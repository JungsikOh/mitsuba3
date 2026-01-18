#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/kdtree.h>
#include <mitsuba/render/sensor.h> // [Fix] Sensor 헤더 추가
#include <drjit/util.h>

#include <fstream>
#include <numeric>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class PhotonMapIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Sensor, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr, Shape)

    // Grid 데이터를 담을 구조체
    struct PhotonGrid {
        UInt32 table_size;       // 해시 테이블 크기 (예: 포톤 수의 1.5~2배)
        Float cell_size;         // 그리드 셀 크기 (검색 반경과 동일하게 설정)
        
        // Grid의 각 Cell이 정렬된 포톤 배열의 어디서 시작하고 끝나는지 저장
        UInt32 cell_start;       
        UInt32 cell_end;

        // 정렬된 포톤 데이터 (원본 데이터를 재배열하여 저장)
        Point3f sorted_p;
        Spectrum sorted_flux;
        Vector3f sorted_wi;
        
        DRJIT_STRUCT(PhotonGrid, table_size, cell_size, cell_start, cell_end, 
                    sorted_p, sorted_flux, sorted_wi)
    };

    // [2] 3D 좌표 -> 1D 해시 인덱스 변환 함수
    UInt32 compute_grid_hash(const Point3i &idx, UInt32 table_size) const {
        // 큰 소수를 이용한 Spatial Hashing (XOR 방식)
        UInt32 x = UInt32(idx.x()) * 73856093u;
        UInt32 y = UInt32(idx.y()) * 19349663u;
        UInt32 z = UInt32(idx.z()) * 83492791u;
        
        return (x ^ y ^ z) % table_size;
    }

    PhotonMapIntegrator(const Properties &props) : Base(props) {
        // [Fix] props.get_int -> props.get<int>
        m_photon_count = props.get<int>("photon_count", 100000);
        m_caustic_photon_count = props.get<int>("caustic_photon_count", 100000);
        m_is_built = false;

    }
    // =============================================================
    //                 Rendering (Main Entry)
    // =============================================================

    TensorXf render(Scene *scene,
                    Sensor *sensor,
                    UInt32 seed,
                    uint32_t spp,
                    bool develop,
                    bool evaluate) override {
        
        if (!m_is_built) {
            Log(Debug, "Building Photon Maps on GPU (Dr.Jit while_loop)...");
            
            // Photon Map 구축 (최초 1회)
            build_global_photon_map(scene);
            build_caustic_photon_map(scene);

            Float global_search_radius = 0.05f;
            Float caustic_search_radius = 0.01f;

            m_global_grid = build_spatial_grid(m_global_count, global_search_radius, m_global_p, m_global_flux, m_global_wi);
            m_caustic_grid = build_spatial_grid(m_caustic_count, caustic_search_radius, m_caustic_p, m_caustic_flux, m_caustic_wi);

            m_is_built = true;
            
            if constexpr (dr::is_jit_v<Float>) {
                dr::eval(m_global_count, m_caustic_count);
                Log(Debug, "Photon Map Build Complete. Global: %i, Caustic: %i", 
                    m_global_count[0], m_caustic_count[0]);

                // save_photon_map_to_ply("global_photons.ply", m_global_p, m_global_flux, m_global_count[0]);
                // save_photon_map_to_ply("caustic_photons.ply", m_caustic_p, m_caustic_flux, m_caustic_count[0]);
            } else {
                Log(Debug, "You can only use 'cuda_ad_rgb'");
            }
        }
        return Base::render(scene, sensor, seed, spp, develop, evaluate);
    }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0)) return { 0.f, false };

        // --------------------- Configure Loop State ----------------------
        struct LoopState {
            Ray3f ray;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            SurfaceInteraction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta; // 이전 바운스가 정반사(거울/유리)였는지 여부
            UInt32 depth;
            Mask valid_ray;
            Bool active;
            Sampler* sampler; // 포인터로 전달

            DRJIT_STRUCT(LoopState, ray, throughput, result, eta, prev_si, prev_bsdf_pdf, prev_bsdf_delta, depth, valid_ray, active, sampler)
        };

        // Variables caching information from the previous bounce
        SurfaceInteraction3f    prev_si               = dr::zeros<SurfaceInteraction3f>();
        Float                   prev_bsdf_pdf         = 1.f;
        Bool                    prev_bsdf_delta       = true;

        LoopState state;
        state.ray = Ray3f(ray_);
        state.throughput = Spectrum(1.f);
        state.result = Spectrum(0.f);
        state.eta = Float(1.f);
        state.prev_si = prev_si;
        state.prev_bsdf_pdf = prev_bsdf_pdf;
        state.prev_bsdf_delta = prev_bsdf_delta; // 첫 레이는 Specular처럼 취급 (직접 발광체 보임)
        state.depth = 0;
        state.valid_ray = (scene->environment() != nullptr);
        state.active = active;
        state.sampler = sampler;

        // --------------------- Dr.Jit While Loop ----------------------
        
        dr::tie(state) = dr::while_loop(
            dr::make_tuple(state),
            [this](const LoopState& ls) {
                return ls.active && (ls.depth < m_max_depth);
            },
            [this, scene](LoopState& ls) {
                
                // Intersection
                SurfaceInteraction3f si = scene->ray_intersect(ls.ray, ls.active);
                ls.active &= si.is_valid(); // 허공이면 종료

                // -----------------------------------------------------------------
                // Emission
                // -----------------------------------------------------------------
                // 조건: 카메라가 직접 보거나(Depth=0) 혹은 거울/유리(Delta)를 통해 본 경우
                // (Diffuse 표면을 통해 본 빛은 아래 Photon Map 단계에서 처리됨)
                if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                    DirectionSample3f ds(scene, si, ls.prev_si);
                    Float em_pdf = 0.f;

                    if (dr::any_or<true>(!ls.prev_bsdf_delta))
                        em_pdf = scene->pdf_emitter_direction(ls.prev_si, ds, !ls.prev_bsdf_delta);

                    // Compute MIS weight for emitter sample from previous bounce
                    Float mis_bsdf = mis_weight(ls.prev_bsdf_pdf, em_pdf);

                    Mask visible_emission = ls.active && !ls.prev_bsdf_delta;

                    Spectrum Le = ds.emitter->eval(si, ls.prev_bsdf_pdf > 0.f) * mis_bsdf;
                    ls.result += dr::select(ls.active, ls.throughput * Le, 0.f);
                }

                // -----------------------------------------------------------------
                // Direct Illumination (NEE: Next Event Estimation)
                // -----------------------------------------------------------------
                // 그림자 레이를 포함한 전체 경로 길이가 m_max_depth를 초과하지 않도록 Mask 변수를 생성.
                Bool active_next = (ls.depth + 1 < m_max_depth) && si.is_valid();

                if (dr::none_or<false>(active_next)) {
                    ls.active = active_next;
                    ls.valid_ray |= (si.emitter(scene) != nullptr);
                    return; // early exit for scalar mode
                }   

                BSDFContext ctx;
                BSDFPtr bsdf = si.bsdf(ls.ray);
                Mask active_nee = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth); // Smooth BSDF만 NEE

                if (dr::any_or<true>(active_nee)) {
                    auto [ds, em_weight] = scene->sample_emitter_direction(si, ls.sampler->next_2d(), true, active_nee);

                    Vector3f wo_world = ds.d;
                    Vector3f wo_local = si.sh_frame.to_local(wo_world);
                        
                    // ------ Evaluate BSDF * cos(theta)
                    auto [bsdf_val, bsdf_pdf] = bsdf->eval_pdf(ctx, si, wo_local, active_nee);
                    bsdf_val = si.to_world_mueller(bsdf_val, -wo_local, si.wi);

                    // Compute the MIS weight
                    Float mis_em = dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                        
                    ls.result += dr::select(active_nee, ls.throughput * em_weight * bsdf_val * mis_em, 0.f);

                }

                // -----------------------------------------------------------------
                // BSDF Sampling & Path Decision (핵심 분기)
                // -----------------------------------------------------------------
                // "다음에 어디로 갈까?"를 먼저 샘플링하여, 그 성격(Diffuse vs Specular)에 따라 전략을 바꿈
                
                auto [bs, bs_weight] = bsdf->sample(ctx, si, ls.sampler->next_1d(), ls.sampler->next_2d(), ls.active);
                
                Mask valid_sample = ls.active && !has_flag(bs.sampled_type, BSDFFlags::Null);

                // 분기 조건: 샘플링된 경로가 Diffuse인가? Specular인가?
                Mask is_diffuse_bounce  = valid_sample && has_flag(bs.sampled_type, BSDFFlags::Smooth);
                Mask is_specular_bounce = valid_sample && has_flag(bs.sampled_type, BSDFFlags::Delta) || has_flag(bs.sampled_type, BSDFFlags::Glossy);

                // =====================================
                // [Case A] Diffuse Indirect -> Photon Map & STOP
                // =====================================
                if (dr::any_or<true>(is_diffuse_bounce)) {
                    // Photon Map 조회 (Global + Caustic)
                    Spectrum pm_radiance = estimate_irradiance_grid(si, m_global_grid);
                    
                    // Caustic Map이 있다면 추가
                    Spectrum caustic_val = Spectrum(0.f);
                    if (dr::any_or<true>(ls.prev_bsdf_delta)) caustic_val = estimate_irradiance_grid(si, m_caustic_grid);
                    pm_radiance += dr::select(ls.prev_bsdf_delta, caustic_val, 0.f);

                    // 결과에 더하고, 이 경로는 여기서 '사망(Terminate)' 처리
                    // (이유: Diffuse 이후의 빛은 PM이 다 가져왔다고 가정)
                    Spectrum rho = bsdf->eval_diffuse_reflectance(si);
                    rho = si.to_world_mueller(rho, -bs.wo, si.wi);
                    ls.result += dr::select(ls.active, ls.throughput * pm_radiance * rho  * dr::InvPi<Float>, 0.f);
                }

                // =====================================
                // [Case B] Specular Indirect -> Continue
                // =====================================
                // 거울/유리인 경우에만 루프를 계속 돈다.
                
                // 다음 루프로 넘어갈 활성 마스크 업데이트
                // "유효한 샘플이고" AND "Specular(Delta) 성분일 때만" 살아남음 + finalGathering을 통해 퀄리티 UP
                Mask continue_path = (valid_sample && is_specular_bounce) || (ls.depth < 3);
                
                // 상태 업데이트 (살아남은 녀석들만)
                bs_weight = si.to_world_mueller(bs_weight, -bs.wo, si.wi);
                ls.throughput = dr::select(continue_path, ls.throughput * bs_weight, ls.throughput);
                ls.eta       *= bs.eta;
                ls.ray        = si.spawn_ray(si.to_world(bs.wo));

                ls.prev_si = SurfaceInteraction3f(si);
                ls.prev_bsdf_pdf = bs.pdf;
                ls.prev_bsdf_delta = has_flag(bs.sampled_type, BSDFFlags::Delta);
                
                ls.depth++;
                ls.active &= continue_path; // Diffuse였던 애들은 여기서 false가 되어 루프 종료

                // -----------------------------------------------------------------
                // Russian Roulette (살아남은 Specular 경로에 대해서만 수행)
                // -----------------------------------------------------------------
                Mask rr_active = ls.active && (ls.depth >= m_rr_depth);
                if (dr::any_or<true>(rr_active)) {
                    Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));
                    Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);
                    Mask rr_continue = ls.sampler->next_1d() < rr_prob;
                    
                    ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));
                    ls.active &= active_next && (!rr_active || rr_continue) && (throughput_max != 0.f);
                }
            }
        );

        return { 
            state.result,
            dr::any(dr::neq(state.result, 0.f)) // 결과가 있으면 유효한 레이로 간주
        };
    }

    // =============================================================
    //                 Photon Mapping Build (GPU)
    // =============================================================

protected:

    void build_global_photon_map(Scene *scene) {
            // [1] Sampler 생성
            ref<Sampler> sampler = static_cast<Sampler*>(PluginManager::instance()->create_object<Sampler>(Properties("independent")));
            // 시드 설정 (웨이브프론트 크기만큼)
            sampler->seed(0, m_photon_count); 

            // [2] 버퍼 준비 (최대 크기)
            size_t capacity = (size_t)m_photon_count * (size_t)m_max_depth;
            
            // 데이터를 저장할 임시 버퍼 (Sparse)
            Point3f buf_p = dr::zeros<Point3f>(capacity);
            Spectrum buf_flux = dr::zeros<Spectrum>(capacity);
            Vector3f buf_wi = dr::zeros<Vector3f>(capacity);
            
            // 해당 슬롯이 채워졌는지 표시할 마스크 버퍼
            UInt32 buf_valid = dr::zeros<UInt32>(capacity);

            // [3] 초기 광선 생성 및 Ray Index 생성
            auto [ray, flux, emitter] = scene->sample_emitter_ray(0.f, sampler->next_1d(), sampler->next_2d(), sampler->next_2d());
            
            // 각 광선의 고유 번호 (0, 1, 2, ... m_photon_count-1)
            UInt32 ray_idx = dr::arange<UInt32>(m_photon_count);

            struct BuildState {
                Ray3f ray;
                Spectrum flux;
                Bool active;
                UInt32 depth;
                UInt32 ray_idx; // 광선 인덱스 추가
                SurfaceInteraction3f si;
                Sampler* sampler;

                DRJIT_STRUCT(BuildState, ray, flux, active, depth, ray_idx, si, sampler)
            };

            BuildState state;
            state.ray = ray;
            state.flux = flux;
            state.active = true;
            state.depth = 0;
            state.ray_idx = ray_idx;
            state.si = dr::zeros<SurfaceInteraction3f>();
            state.sampler = sampler.get();

            // [4] 루프 실행
            dr::tie(state) = dr::while_loop(
                dr::make_tuple(state),
                [this](const BuildState& s) { return s.active && (s.depth < m_max_depth); },
                [this, scene, &buf_p, &buf_flux, &buf_wi, &buf_valid](BuildState& s) {
                    s.si = scene->ray_intersect(s.ray, s.active);
                    s.active &= s.si.is_valid();

                    if (dr::none_or<false>(s.active)) return;

                    BSDFPtr bsdf = s.si.bsdf();
                    UInt32 flags = bsdf->flags();
                    Mask is_specular = has_flag(flags, BSDFFlags::Glossy) || has_flag(flags, BSDFFlags::Delta);
                    Mask is_diffuse = !is_specular;

                    // 글로벌 포톤맵에 저장할 조건
                    Mask valid_global = s.active && is_diffuse && (s.depth > 0);

                    if (dr::any_or<true>(valid_global)) {
                        // [핵심 변경] Atomic Add 대신 결정론적 인덱스 계산
                        // 충돌 없이 자기만의 자리를 찾아갑니다.
                        UInt32 slot = s.ray_idx * m_max_depth + s.depth;
                        
                        dr::scatter(buf_p, s.si.p, slot, valid_global);
                        dr::scatter(buf_flux, s.flux, slot, valid_global);
                        dr::scatter(buf_wi, s.ray.d, slot, valid_global);
                        
                        // 유효하다는 마킹
                        dr::scatter(buf_valid, UInt32(1), slot, valid_global);
                    }
                    
                    // ... (BSDF 샘플링 및 다음 경로 추적 로직은 기존 동일) ...
                    // (생략 없이 기존 코드 그대로 유지하시면 됩니다)
                    BSDFContext ctx;
                    auto [bs_sample, bs_weight] = bsdf->sample(ctx, s.si, s.sampler->next_1d(), s.sampler->next_2d(), s.active);
                    bs_weight = s.si.to_world_mueller(bs_weight, -bs_sample.wo, s.si.wi);
                    
                    s.flux *= bs_weight;
                    s.ray = s.si.spawn_ray(s.si.sh_frame.to_world(bs_sample.wo));
                    s.active &= dr::any(unpolarized_spectrum(s.flux) > 0.f);
                    s.depth++;
                }
            );

            // [5] 데이터 압축 (Compaction)
            // 1. buf_valid != 0 조건을 통해 Mask로 변환 후 compress에 전달
            // 2. 결과(인덱스들)는 UInt32 타입으로 저장
            if constexpr (dr::is_jit_v<Float>) {
                // JIT 모드에서만 압축 및 Gather 실행
                UInt32 valid_indices = dr::compress(dr::neq(buf_valid, 0u));
                size_t count = dr::width(valid_indices);
                m_global_count = dr::opaque<UInt32>((uint32_t)count); 
                
                buf_flux /= Float(m_photon_count);

                m_global_p = dr::gather<Point3f>(buf_p, valid_indices);
                m_global_flux = dr::gather<Spectrum>(buf_flux, valid_indices);
                m_global_wi = dr::gather<Vector3f>(buf_wi, valid_indices);
                
            } else {
                m_global_count = 0;
            }
        }


    void build_caustic_photon_map(Scene *scene) {
        BoundingBox3f bbox = scene->bbox();
        Point3f target_center = bbox.center();
        Float target_radius = dr::norm(bbox.max - bbox.min) * 0.5f;
        Mask has_target = bbox.valid();

        // if (dr::any(has_target)) {
        //     m_caustic_count = dr::zeros<UInt32>(1);
        //     return;
        // }

        // [Fix] create_object<Sampler>
        ref<Sampler> sampler = static_cast<Sampler*>(PluginManager::instance()->create_object<Sampler>(Properties("independent")));
        sampler->seed(0, m_caustic_photon_count);

        std::vector<ref<Emitter>> emitters = scene->emitters();
    
        ScalarFloat total_weight = 0.f;
        std::vector<ScalarFloat> weights;

        for (ref<Emitter> emitter : emitters) {
            ScalarFloat w = emitter.get()->sampling_weight();
            weights.push_back(w);

            total_weight += w;
        }
        
        // Normalize weights
        if (total_weight > 0.f) {
            ScalarFloat inv_total = 1.f / total_weight; // 나눗셈보다 곱셈이 빠르므로 역수를 구함
            
            for (ScalarFloat &w : weights) {
                w *= inv_total;
            }
        } else {
            // 만약 모든 가중치가 0이라면 (광원이 없거나 다 꺼짐),
            Log(Warn, "Total emitter weight is zero!");
        }

        std::vector<ScalarInt32> photon_counts;
        photon_counts.reserve(weights.size());

        for (ScalarFloat w : weights) {
            photon_counts.push_back(static_cast<ScalarInt32>(m_caustic_photon_count * w));
        }

        // [2] 버퍼 준비 (최대 크기)
        size_t capacity = (size_t)m_caustic_photon_count * (size_t)m_max_depth;
            
        // 데이터를 저장할 임시 버퍼 (Sparse)
        Point3f buf_p = dr::zeros<Point3f>(capacity);
        Spectrum buf_flux = dr::zeros<Spectrum>(capacity);
        Vector3f buf_wi = dr::zeros<Vector3f>(capacity);

        UInt32 global_offset = 0;

            
        // 해당 슬롯이 채워졌는지 표시할 마스크 버퍼
        UInt32 buf_valid = dr::zeros<UInt32>(capacity);

        for (size_t i = 0; i < emitters.size(); ++i) {
            ScalarInt32 count = photon_counts[i];
            if (count <= 0) continue;

            auto [ray, flux] = emitters[i].get()->sample_ray(0.f, sampler->next_1d(), sampler->next_2d(), sampler->next_2d());

            // Vector3f surface_to_emitter = target_center - emitter_pos.p;
            // Float dist = dr::norm(surface_to_emitter);
            // surface_to_emitter /= dist;

            // Float sin_theta = dr::minimum(1.0f, target_radius / dist);
            // Float cos_theta_max = dr::sqrt(1.0f - sin_theta * sin_theta);

            // Vector3f dir_local = warp::square_to_uniform_cone(sampler.get()->next_2d(), cos_theta_max);
            // Vector3f dir_world = Frame3f(surface_to_emitter).to_world(dir_local);

            // Ray3f light_ray = Ray3f(emitter_pos.p, dir_world);
            
            struct BuildState {
                Ray3f ray;
                Spectrum flux;
                Bool active;
                UInt32 depth;
                UInt32 ray_idx;
                Bool has_hit_specular;
                SurfaceInteraction3f si;
                Sampler* sampler;
                
                DRJIT_STRUCT(BuildState, ray, flux, active, depth, ray_idx, has_hit_specular, si, sampler)
            };

            BuildState state;
            state.ray = ray;
            state.flux = flux;
            state.active = true;
            state.depth = 0;
            state.ray_idx = dr::arange<UInt32>(count) + global_offset;
            state.has_hit_specular = false;
            state.si = dr::zeros<SurfaceInteraction3f>();
            state.sampler = sampler.get();

            dr::tie(state) = dr::while_loop(
                dr::make_tuple(state),
                [this](const BuildState& bs) { return bs.active && (bs.depth < m_max_depth); },
                [this, scene, &buf_p, &buf_flux, &buf_wi, &buf_valid](BuildState& bs) { // 선언한 변수들 중에 가져올 것들을 입력.
                    bs.si = scene->ray_intersect(bs.ray, bs.active);
                    bs.active &= bs.si.is_valid();
                    
                    BSDFPtr bsdf = bs.si.shape->bsdf();
                    UInt32 bsdf_flags = bsdf->flags();

                    Mask is_specular = has_flag(bsdf_flags, BSDFFlags::Glossy) || has_flag(bsdf_flags, BSDFFlags::Delta);
                    Mask is_diffuse = !is_specular;

                    Mask valid_caustic = bs.active && is_diffuse && bs.has_hit_specular;
                    
                    if (dr::any_or<true>(valid_caustic)) {
                        // [저장] 결정론적 인덱싱 (충돌 방지)
                        UInt32 slot = bs.ray_idx * m_max_depth + bs.depth;

                        dr::scatter(buf_p, bs.si.p, slot, valid_caustic);
                        dr::scatter(buf_flux, bs.flux, slot, valid_caustic);
                        dr::scatter(buf_wi, bs.ray.d, slot, valid_caustic);
                        
                        // 유효함 표시 (1)
                        dr::scatter(buf_valid, UInt32(1), slot, valid_caustic);
                        bs.active &= !valid_caustic;
                    }

                    bs.has_hit_specular |= is_specular;
                    
                    BSDFContext ctx;
                    auto [bs_sample, bs_weight] = bsdf->sample(ctx, bs.si, bs.sampler->next_1d(), bs.sampler->next_2d(), bs.active);
                    bs_weight = bs.si.to_world_mueller(bs_weight, -bs_sample.wo, bs.si.wi);

                    bs.flux *= bs_weight;
                    bs.ray = bs.si.spawn_ray(bs.si.sh_frame.to_world(bs_sample.wo));
                    
                    bs.active &= dr::any(unpolarized_spectrum(bs.flux) > 0.f);

                    bs.depth++;
                }
            );

            global_offset += count;

            // [5] 데이터 압축 (Compression) - 모든 루프 종료 후 한 번에 수행
            if constexpr (dr::is_jit_v<Float>) {
                // buf_valid가 0이 아닌(데이터가 있는) 인덱스만 추출
                UInt32 valid_indices = dr::compress(dr::neq(buf_valid, 0u));
                size_t final_count = dr::width(valid_indices);
                buf_flux /= Float(m_caustic_photon_count);

                // Gather로 모으기
                m_caustic_p = dr::gather<Point3f>(buf_p, valid_indices);
                m_caustic_flux = dr::gather<Spectrum>(buf_flux, valid_indices);
                m_caustic_wi = dr::gather<Vector3f>(buf_wi, valid_indices);
                
                m_caustic_count = dr::opaque<UInt32>((uint32_t)final_count);
            } else {
                m_caustic_count = 0;
            }

        }
    }

    PhotonGrid build_spatial_grid(UInt32 count, Float radius, 
                                const Point3f &p, const Spectrum &flux, const Vector3f &wi) {
        
        PhotonGrid grid;
        size_t n_photons;
        if constexpr (dr::is_jit_v<Float>) {
            n_photons = count[0];
        } else {
            n_photons = count;
        }
        
        // 테이블 크기 설정
        size_t table_size = n_photons * 2; 
        grid.table_size = UInt32(table_size);
        grid.cell_size = radius;

        // [1] GPU: Grid Index 및 Hash 계산
        Point3i grid_idx = Point3i(dr::floor(p / grid.cell_size));
        UInt32 hash = compute_grid_hash(grid_idx, grid.table_size);

        // ----------------------------------------------------------------
        // [2] CPU Fallback Sort
        // ----------------------------------------------------------------
        
        // 2-1. GPU 계산 결과 확정 및 동기화 (필수)
        dr::eval(hash);
        dr::sync_thread(); 

        // 2-2. Hash 데이터를 CPU로 복사
        std::vector<uint32_t> cpu_hash(n_photons);
        // Dr.Jit 배열 데이터를 포인터로 접근하여 복사
        dr::store(cpu_hash.data(), hash);

        // 2-3. 정렬을 위한 인덱스 배열 생성 (0, 1, 2, ... n-1)
        std::vector<uint32_t> cpu_indices(n_photons);
        std::iota(cpu_indices.begin(), cpu_indices.end(), 0);

        // 2-4. CPU에서 정렬 수행 (Hash 값을 기준으로 Index 정렬)
        // std::stable_sort를 사용하여 같은 해시 내의 순서를 유지 (선택사항)
        std::stable_sort(cpu_indices.begin(), cpu_indices.end(), 
            [&cpu_hash](uint32_t i1, uint32_t i2) {
                return cpu_hash[i1] < cpu_hash[i2];
            }
        );

        // 2-5. 정렬된 결과를 다시 GPU로 업로드
        UInt32 sort_idx = dr::load<UInt32>(cpu_indices.data(), n_photons);
        
        // GPU상에서 정렬된 Hash 값도 필요하므로 다시 Gather
        // (CPU에서 만든 hash를 다시 올리는 것보다 gather가 나을 수 있음)
        UInt32 sorted_hash = dr::gather<UInt32>(hash, sort_idx);

        // ----------------------------------------------------------------
        // [3] GPU: 데이터 재배열 (Gather)
        // ----------------------------------------------------------------
        
        // 이제 sort_idx가 있으므로 기존 로직대로 진행 가능
        grid.sorted_p    = dr::gather<Point3f>(p, sort_idx);
        grid.sorted_flux = dr::gather<Spectrum>(flux, sort_idx);
        grid.sorted_wi   = dr::gather<Vector3f>(wi, sort_idx);

        // [4] Cell Start / End 테이블 생성 (Scatter Reduce)
        grid.cell_start = dr::full<UInt32>(0xFFFFFFFFu, table_size);
        grid.cell_end   = dr::zeros<UInt32>(table_size);

        // 0, 1, 2... 순차 인덱스 (정렬된 배열 기준)
        UInt32 loop_idx = dr::arange<UInt32>(n_photons); 

        // ReduceOp::Min/Max를 사용하여 시작/끝 지점 기록
        // 주의: dr::ReduceOp 대신 ReduceOp 사용 (네임스페이스 문제 해결)
        dr::scatter_reduce(ReduceOp::Min, grid.cell_start, loop_idx,     sorted_hash);
        dr::scatter_reduce(ReduceOp::Max, grid.cell_end,   loop_idx + 1, sorted_hash);

        return grid;
    }

    Spectrum estimate_irradiance_grid(const SurfaceInteraction3f &si, 
                                   const PhotonGrid &grid) const {
    
        Spectrum irradiance_sum = Spectrum(0.f);
        Float r_sqr = grid.cell_size * grid.cell_size;
        Float inv_r_sqr = 1.0f / r_sqr;

        // [1] 쉐이딩 포인트의 그리드 좌표
        Point3i center_idx = Point3i(dr::floor(si.p / grid.cell_size));

        // [2] 3x3x3 이웃 셀 탐색
        for (int z = -1; z <= 1; ++z) {
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    
                    Point3i neighbor_idx = center_idx + Vector3i(x, y, z);
                    UInt32 bucket_hash = compute_grid_hash(neighbor_idx, grid.table_size);

                    UInt32 start_idx = dr::gather<UInt32>(grid.cell_start, bucket_hash);
                    UInt32 end_idx   = dr::gather<UInt32>(grid.cell_end, bucket_hash);

                    Mask valid_bucket = (start_idx != 0xFFFFFFFFu);
                    if (dr::none_or<false>(valid_bucket)) continue;

                    // [3] 해당 버킷 내의 포톤들만 순회
                    struct LoopState {
                        UInt32 current_idx;
                        Spectrum accumulated_irradiance;
                        Bool active;
                        DRJIT_STRUCT(LoopState, current_idx, accumulated_irradiance, active)
                    };

                    LoopState state;
                    state.current_idx = start_idx;
                    state.accumulated_irradiance = Spectrum(0.f);
                    state.active = valid_bucket;

                    dr::tie(state) = dr::while_loop(
                        dr::make_tuple(state),
                        [end_idx](const LoopState& s) {
                            return s.active && (s.current_idx < end_idx);
                        },
                        [this, &si, r_sqr, inv_r_sqr, &grid](LoopState& s) {
                            // 포톤 데이터 가져오기
                            Point3f p_photon     = dr::gather<Point3f>(grid.sorted_p, s.current_idx);
                            Spectrum flux_photon = dr::gather<Spectrum>(grid.sorted_flux, s.current_idx);
                            Vector3f wi_photon   = dr::gather<Vector3f>(grid.sorted_wi, s.current_idx);

                            Float dist2 = dr::squared_norm(p_photon - si.p);
                            Mask in_radius = dist2 < r_sqr;

                            if (dr::any_or<true>(in_radius)) {
                                // [FIX 1] Normal 검증 추가 (Light Leaking 방지)
                                Float wi_dot_n = dr::dot(wi_photon, si.n);
                                Mask valid_direction = wi_dot_n > 0.f; // 위쪽에서 들어오는 빛만
                                
                                // [FIX 2] Cone Filter (Simpson's Kernel) 적용
                                Float sqr_term = 1.0f - dist2 * inv_r_sqr;
                                Float weight = sqr_term * sqr_term; // (1 - d²/r²)²
                                
                                // [FIX 3] BSDF 평가 제거 (Irradiance만 추정)
                                Mask valid_photon = in_radius && valid_direction;
                                Spectrum contribution = dr::select(in_radius, flux_photon * weight, 0.f);
                                
                                s.accumulated_irradiance += contribution;
                            }
                            s.current_idx++;
                        }
                    );
                    
                    irradiance_sum += state.accumulated_irradiance;
                }
            }
        }
        
        // [FIX 4] Simpson Kernel 정규화: 3/(πr²)
        Float normalization = 3.0f * dr::InvPi<Float> * inv_r_sqr;
        return irradiance_sum * normalization;
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    std::string to_string() const override {
        return tfm::format("PhotonMapIntegrator[\n"
            "  max_depth = %i,\n"
            "  photon_count = %i\n"
            "  caustic_photon_count = %i\n"
            "]", m_max_depth, m_photon_count, m_caustic_photon_count);
    }

    void save_photon_map_to_ply(const std::string &filename, 
                            const Point3f &p, 
                            const Spectrum &flux, 
                            uint32_t count) const {
        
        Log(Info, "Saving photon map to '%s' (%i photons)...", filename.c_str(), count);

        std::ofstream file(filename);
        if (!file.fail()) {
            // 1. PLY 헤더 작성
            file << "ply\n";
            file << "format ascii 1.0\n";
            file << "element vertex " << count << "\n";
            file << "property float x\n";
            file << "property float y\n";
            file << "property float z\n";
            file << "property uchar red\n";
            file << "property uchar green\n";
            file << "property uchar blue\n";
            file << "end_header\n";

            // 2. GPU 데이터를 CPU로 동기화
            dr::eval(p, flux);
            dr::sync_thread();

            // 3. 데이터 쓰기
            for (uint32_t i = 0; i < count; ++i) {
                // Dr.Jit 배열에서 스칼라 값 추출
                Point3f pos_val = dr::slice(p, i);
                Spectrum flux_val = dr::slice(flux, i);

                // JIT 모드에서는 eval 필요
                if constexpr (dr::is_jit_v<Float>) {
                    dr::eval(pos_val, flux_val);
                    dr::sync_thread();
                }

                // CPU로 값 읽기
                float px = pos_val.x()[0];
                float py = pos_val.y()[0];
                float pz = pos_val.z()[0];

                // Spectrum -> RGB 변환 및 톤매핑
                float scale = 100.0f;
                
                // Spectrum 채널 추출 (RGB 또는 Spectral에 따라 다름)
                float r_val = 0.f, g_val = 0.f, b_val = 0.f;
                
                if constexpr (dr::is_jit_v<Float>) {
                    // GPU 배열에서 스칼라 값으로 변환
                    auto flux_scalar = flux_val;
                    dr::eval(flux_scalar);
                    dr::sync_thread();
                    
                    // RGB 채널 접근 (Spectrum이 RGB인 경우)
                    r_val = flux_scalar[0][0];
                    g_val = (Spectrum::Size >= 2) ? flux_scalar[1][0] : flux_scalar[0][0];
                    b_val = (Spectrum::Size >= 3) ? flux_scalar[2][0] : flux_scalar[0][0];
                } else {
                    // Scalar 모드
                    r_val = flux_val[0];
                    g_val = (Spectrum::Size >= 2) ? flux_val[1] : flux_val[0];
                    b_val = (Spectrum::Size >= 3) ? flux_val[2] : flux_val[0];
                }

                // 톤매핑 및 클램핑
                r_val = std::min(1.0f, std::max(0.0f, r_val * scale));
                g_val = std::min(1.0f, std::max(0.0f, g_val * scale));
                b_val = std::min(1.0f, std::max(0.0f, b_val * scale));

                // 감마 보정
                r_val = std::pow(r_val, 1.f / 2.2f);
                g_val = std::pow(g_val, 1.f / 2.2f);
                b_val = std::pow(b_val, 1.f / 2.2f);

                // RGB를 0-255 범위로 변환
                int r_int = static_cast<int>(r_val * 255.0f);
                int g_int = static_cast<int>(g_val * 255.0f);
                int b_int = static_cast<int>(b_val * 255.0f);

                file << px << " " << py << " " << pz << " "
                     << r_int << " " << g_int << " " << b_int << "\n";
            }
            file.close();
            Log(Info, "Saved PLY successfully.");
        } else {
            Log(Warn, "Could not open file '%s' for writing!", filename.c_str());
        }
    }

    MI_DECLARE_CLASS(PhotonMapIntegrator)

protected:
    UInt32 m_final_gathering_depth = 3;
    int m_photon_count;
    int m_caustic_photon_count;
    bool m_is_built;

    UInt32 m_global_count = dr::zeros<UInt32>(1);
    Point3f m_global_p;
    Spectrum m_global_flux;
    Vector3f m_global_wi;

    UInt32 m_caustic_count = dr::zeros<UInt32>(1);
    Point3f m_caustic_p;
    Spectrum m_caustic_flux;
    Vector3f m_caustic_wi;

    PhotonGrid m_global_grid;
    PhotonGrid m_caustic_grid;
};

MI_EXPORT_PLUGIN(PhotonMapIntegrator)
NAMESPACE_END(mitsuba)