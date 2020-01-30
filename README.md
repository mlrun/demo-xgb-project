# MLRun Project and GitOps demo

Demo the work of MLRun with Github based projects and automated CI/CD

## Running the demo

in a client or notebook properly configured with MLRun and KubeFlow use the following lines:

```python
from mlrun import load_project

# load the project from GitHub
url = 'git://github.com/mlrun/webhook-demo.git'
proj = load_project('/tmp/myproj', url)

print(proj.to_yaml())

# run the project main pipeline (build, data prep, train, deploy model)
pipeline = proj.run(arguments={}, artifacts_path='v3io:///users/admin/mlrun/kfp/{{workflow.uid}}/')
```

## Files

* [Project spec (functions, workflows, etc)](project.yaml)
* [Local function spec (XGboost)](function.yaml)
* [Function code](iris.py)
* [Workflow code (init + dsl)](workflow.py)


## Pipeline

<br><p align="center"><img src="./pipeline.PNG" width="500"/></p><br>
