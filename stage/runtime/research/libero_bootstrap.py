"""Explicit LIBERO paths and per-episode config; no global environment edits."""
import os
from pathlib import Path
import sys
from research.common import read_json


def prepare(path):
    spec=read_json(path)
    dependency=Path(spec['dependency_root'])
    data_root=Path(os.environ.get('STAGE_DATA_ROOT', str(dependency)))
    stage_root=Path(spec['source_root']).parent
    paths=[spec['source_root'],str(stage_root),spec['backend_root'],spec['libero_pro_source'],
           str(Path(spec['backend_root'])/'libero/dependencies/python311'),str(dependency/'vendor'),
           spec.get('robosuite_source', os.environ.get('STAGE_ROBOSUITE_SOURCE', str(dependency/'sources/robosuite'))),
           str(dependency/'sources/rlinf'),str(dependency/'sources/openpi/src'),
           str(dependency/'sources/openpi/packages/openpi-client/src')]
    extra_deps = os.environ.get('STAGE_EXTRA_DEPS')
    if extra_deps:
        paths.append(extra_deps)
    sys.path[:0]=paths
    sys.path.extend([str(dependency/'jax_overlay')])
    output=Path(spec['output_dir'])
    config=output/'private/libero_config'
    config.mkdir(parents=True,exist_ok=True)
    core=Path(spec['libero_pro_source'])/'liberopro/liberopro'
    values=dict(benchmark_root=core,bddl_files=spec['bddl_root'],init_states=spec['init_states_root'],
                datasets=output/'unused_training_data',assets=core/'assets')
    target=config/'config.yaml'
    tmp=target.with_name(f'config.{os.getpid()}.tmp')
    tmp.write_text(''.join(f'{key}: {value}\n' for key,value in values.items()))
    tmp.replace(target)
    os.environ.update(LIBERO_CONFIG_PATH=str(config),LIBERO_TYPE='pro',ROBOT_PLATFORM='LIBERO',
        LIBERO_PRO_ASSET_PATH=os.environ.get('LIBERO_PRO_ASSET_PATH', str(core/'assets')),
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',JAX_PLATFORMS='cpu',
        OPENPI_DATA_HOME=os.environ.get('OPENPI_DATA_HOME', str(data_root/'openpi_cache')),
        PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',NCCL_IB_DISABLE='1',TORCHINDUCTOR_COMPILE_THREADS='1')
    return spec
