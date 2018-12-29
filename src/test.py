from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask.ext.jsonpify import jsonify

app = Flask(__name__)
api = Api(app)


class Employees(Resource):
    def get(self):
        return {'employees'}


api.add_resource(Employees, '/employees')

if __name__ == '__main__':
    app.run(port='5002')