#!/usr/bin/env python3
from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, UnitData
from base.model import DataRunner, InertialNetworkData, ModelLoader


def main():
    args = DatasetArgsParser()
    args.parser.add_argument("-m", "--models", nargs="+", help="模型")
    args.parser.add_argument("--models_path", type=str, help="模型文件夹")
    args.parser.add_argument("--using_ahrs", action="store_true", default=False)
    args.parse()

    models = args.args.models
    if models is None or len(models) == 0:
        models = ["model_mi_hw_1128"]

    models_path = "Models"
    if args.args.models_path is not None:
        models_path = args.args.models_path
    loader = ModelLoader(models_path)

    if args.unit:
        # 数据
        data = UnitData(args.unit)
        Data = InertialNetworkData.set_step(20)
        runner = DataRunner(data, Data, using_gt=not args.args.using_ahrs)
        runner.predict_batch(loader.get_by_names(models))

    if args.dataset:
        dataset_path = args.dataset
        datas = DeviceDataset(dataset_path)
        for data in datas:
            runner = DataRunner(data, InertialNetworkData.set_step(10))
            runner.predict_batch(loader.get_by_names(models))


if __name__ == "__main__":
    main()
