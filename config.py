
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app, support_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")
