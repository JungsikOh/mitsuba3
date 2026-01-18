import os
import sys
import mitsuba as mi

# 1. Mitsuba Variant 설정 (GPU 모드 필수)
mi.set_variant('cuda_ad_rgb')

mi.set_log_level(mi.LogLevel.Debug)

photonmapper = {
    'type': 'photonmapper',  # C++ 플러그인 이름
    'photon_count': 500_000,
    'caustic_photon_count': 2_000_000,
    'max_depth': 30,
}

path = {
    'type': 'path',
    'max_depth': 15,
}

# 4. 씬 로드 및 렌더링
print("Loading scene...")
my_integrator = mi.load_dict(photonmapper)
#  mi.load_file("scenes/water-caustic/scene_v0.6.xml")
#  mi.load_file("scenes/veach-ajar/scene_v3.xml")
scene = mi.load_file("scenes/water-caustic/scene_v0.6.xml")
print("Rendering...")
# render() 호출 시 C++ 코드 내부의 render() -> build_global_... -> sample() 순서로 실행됨
image = mi.render(scene, integrator=my_integrator, spp=256)

# 5. 결과 저장
print("Saving result...")
mi.util.write_bitmap("scenes/water-caustic/outputs/photon_map_photon_0.2M.png", image)
print("Done!")