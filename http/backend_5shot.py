from flask import Flask, jsonify, request
from pathlib import Path


app = Flask(__name__, static_url_path='')


@app.route('/')
def index():
    return app.send_static_file("index_5shot.html")


@app.route('/api/experiments')
def get_experiments():
    all_exps = (Path(__file__).parent / "static").glob("[0-9]*_*_5shot_*")
    exps = [str(x.name) for x in sorted(list(all_exps), key=lambda x: int(str(x.name).split("_")[0]))]
    return jsonify(exps)

@app.route('/api/samples')
def get_samples():
    all_exps = (Path(__file__).parent / "static").glob("[0-9]*_*_5shot_*")
    exps = [str(x.name) for x in sorted(list(all_exps))]

    name = request.args.get("exp")
    if name in exps:
        all_samples = (Path(__file__).parent / "static" / name).glob("[0-9]*_[0-8]*")
        samples = [str(x.name) for x in sorted(list(all_samples))]
    else:
        samples = []
    return jsonify(samples)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=17003)
