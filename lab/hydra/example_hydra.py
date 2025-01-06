import hydra
from omegaconf import DictConfig


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def my_app(cfg: DictConfig) -> None:
    print(f'Hello, {cfg.name}!')


if __name__ == '__main__':
    my_app()
