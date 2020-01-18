rom kfp import dsl

artifacts_path = './'
funcs = {}

xgb = funcs['xgb']
serving = funcs['serving']
df_path = 'mydf.csv'

@dsl.pipeline(
    name='My XGBoost training pipeline',
    description='Shows how to use mlrun.'
)
def kfpipeline(
        eta=[0.1, 0.2, 0.3], gamma=[0.1, 0.2, 0.3]
):
    builder = xgb.deploy_step(with_mlrun=False)

    ingest = xgb.as_step(name='ingest_iris', handler='iris_generator',
                           image=builder.outputs['image'],
                           params={'target': df_path},
                           outputs=['iris_dataset'], out_path=artifacts_path)

    train = xgb.as_step(name='xgb_train', handler='xgb_train',
                          image=builder.outputs['image'],
                          hyperparams={'eta': eta, 'gamma': gamma},
                          selector='max.accuracy',
                          inputs={'dataset': ingest.outputs['iris_dataset']},
                          outputs=['model'], out_path=artifacts_path)

    deploy = serving.deploy_step(models={'iris_v1': train.outputs['model']})
