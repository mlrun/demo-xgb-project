from kfp import dsl
from mlrun import mount_v3io

artifacts_path = './'
funcs = {}

df_path = 'mydf.csv'


def init_functions(functions: dict, params=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())


@dsl.pipeline(
    name='My XGBoost training pipeline',
    description='Shows how to use mlrun.'
)
def kfpipeline(
        eta=[0.1, 0.2, 0.3], gamma=[0.1, 0.2, 0.3]
):
    # first step build the function container
    builder = funcs['xgb'].deploy_step(with_mlrun=False)

    # use xgb.iris_generator function to generate data (container image from the builder)
    ingest = funcs['xgb'].as_step(name='ingest_iris', handler='iris_generator',
        image=builder.outputs['image'],
        params={'target': df_path},
        outputs=['iris_dataset'], out_path=artifacts_path)

    # use xgb.xgb_train function to train on the data (from the generator step)
    train = funcs['xgb'].as_step(name='xgb_train', handler='xgb_train',
        image=builder.outputs['image'],
        hyperparams={'eta': eta, 'gamma': gamma},
        selector='max.accuracy',
        inputs={'dataset': ingest.outputs['iris_dataset']},
        outputs=['model'], out_path=artifacts_path)

    # deploy the trained model using a nuclio real-time function
    deploy = funcs['serving'].deploy_step(models={'iris_v1': train.outputs['model']})
