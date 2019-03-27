import sys
sys.path.append("../")

import argparse
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin

from main import MainPipeline, Config
from planner.naive_planner import NaivePlanner
from scorer.global_direction import GlobalDirectionExpert
from scorer.product_of_experts import WeightedProductOfExperts
from scorer.relation_direction import RelationDirectionExpert
from scorer.relation_transitions import RelationTransitionsExpert
from scorer.splitting_tendencies import SplittingTendenciesExpert
from utils.delex import concat_entity
from utils.graph import Graph
from data.WebNLG.reader import WebNLGDataReader

server_config = {
    "port": 5001,
    "reader": WebNLGDataReader,
    "planner": NaivePlanner(WeightedProductOfExperts([
        RelationDirectionExpert,
        GlobalDirectionExpert,
        SplittingTendenciesExpert,
        RelationTransitionsExpert
    ]))
}

dataset_name = server_config["reader"].DATASET
main_config = Config(reader=server_config["reader"], planner=server_config["planner"])

base_path = os.path.dirname(os.path.abspath(__file__))


def server(pipeline_res, host, port, debug=True):
    app = Flask(__name__)
    CORS(app)

    # @app.route('/', methods=['GET'])
    # @app.route('/index.html', methods=['GET'])
    # def root():
    #     print("got to root")
    #     return app.send_static_file('static/index.html')

    @app.route('/graphs', methods=['GET'])
    @cross_origin()
    def graphs():
        data = [d.graph.as_rdf() for d in pipeline_res["pre-process"]["train"].data]
        return jsonify(data)

    @app.route('/plans/<type>', methods=['POST'])
    @cross_origin()
    def plans(type):
        triplets = request.get_json(force=True)
        graph = Graph(triplets)

        all_plans = graph.exhaustive_plan() if type == "full" else graph.plan_all()

        return jsonify({
            "concat": {n: concat_entity(n) for n in graph.nodes},
            "linearizations": [{"l": l.replace("[", " [ ").replace("]", " ] ").replace("  ", " ")}
                               for l in all_plans.linearizations()]
        })

    @app.route('/translate', methods=['POST'])
    @cross_origin()
    def translate():
        plans = request.get_json(force=True)
        model = pipeline_res["train-model"]

        return jsonify(model.translate(plans))

    @app.route('/', defaults={"filename": "index.html"})
    @app.route('/main.js', defaults={"filename": "main.js"})
    @app.route('/style.css', defaults={"filename": "style.css"})
    def serve_static(filename):
        print("Serving static", filename, os.path.join(base_path, 'static'), filename)
        return send_from_directory(os.path.join(base_path, 'static'), filename)

    app.run(debug=debug, host=host, port=port, use_reloader=False, threaded=True)


if __name__ == "__main__":
    res = MainPipeline.mutate({"config": main_config}).execute(dataset_name, cache_name=dataset_name)

    parser = argparse.ArgumentParser(description="Chimera REST Server")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5001")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    server(res, host=args.ip, port=args.port, debug=args.debug)
