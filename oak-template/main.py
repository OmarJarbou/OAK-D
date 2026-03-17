import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pathlib import Path

import depthai as dai
from utils.snaps_producer import SnapsProducer
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork



model = "luxonis/yolov6-nano:r2-coco-512x288"

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device()

api_key = os.getenv("OAK_API_KEY")


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    model_description = dai.NNModelDescription(model)
    platform = device.getPlatformAsString()
    model_description.platform = platform
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(
            model_description,
            apiKey=api_key,
        )
    )

    input_node = pipeline.create(dai.node.Camera).build()

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive
    )

    visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Visualizations", nn_with_parser.out, "images")

    snaps_producer = pipeline.create(SnapsProducer).build(
        nn_with_parser.passthrough,
        nn_with_parser.out,
        label_map=nn_archive.getConfigV1().model.heads[0].metadata.classes,
    )
    

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break