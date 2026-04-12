from roboflow import Roboflow

#https://universe.roboflow.com/project-qynru/can-gcc8l


rf = Roboflow(api_key="PR0Ro7UzkELszzzyaDBg")  # free account at roboflow.com
project = rf.workspace("project-qynru").project("can-gcc8l")
version = project.version(1)
dataset = version.download("yolov8")
