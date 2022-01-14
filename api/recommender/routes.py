import time

from flask import Blueprint, g, jsonify, request, render_template

from api.exceptions import NoClass
from utils.misc import get_traceback, logger

from .control import REControl

blue_print = Blueprint('recommender', __name__, url_prefix='/recommender')

@blue_print.route('/home')
def home():
    return render_template('home.html')



@blue_print.route('/get-food-recommendations/<user_id>', methods=['GET'])
def get_food_recommendations(user_id):
	start_time = time.time()

	N = int(request.args.get('N', 10))
	model_name = request.args.get('model', 'lsa').lower()
	model_version = request.args.get('version', 'v1.0.0')

	try:
		recommender_engine = REControl(g.db_main, g.db_ai, g.fs_ai)
		response = recommender_engine.get_food_recommendations(model_name, model_version, int(user_id), int(N))
	except NoClass as e:
		logger('FOODPIE_RECOMMENDER', 'ERR', get_traceback(e))
		logger('FOODPIE_RECOMMENDER', 'ERR', e.__str__())
		return jsonify({'message': e.__str__()}), e.http_status()
	except Exception as e:
		logger('FOODPIE_RECOMMENDER', 'ERR', get_traceback(e))
		return jsonify({'message': 'Unknown error! Please try after sometime.'}), 500

	end_time = time.time()
	logger('FOODPIE_RECOMMENDER', 'EXE_TIME', 'Execution time of get_food_recommendations() for user_id={} with N={} is {}.'.format(user_id, N, end_time-start_time))
	return jsonify(response), 200

@blue_print.route('/update-model/<model_name>/<model_version>', methods=['GET'])
def update_model(model_name, model_version):
	start_time = time.time()

	try:
		recommender_engine = REControl(g.db_main, g.db_ai, g.fs_ai)
		response = recommender_engine.update_model(model_name, model_version)
	except NoClass as e:
		logger('FOODPIE_RECOMMENDER', 'ERR', get_traceback(e))
		logger('FOODPIE_RECOMMENDER', 'ERR', e.__str__())
		return jsonify({'message': e.__str__()}), e.http_status()
	except Exception as e:
		logger('FOODPIE_RECOMMENDER', 'ERR', get_traceback(e))
		return jsonify({'message': 'Unknown error! Please try after sometime.'}), 500

	end_time = time.time()
	logger('FOODPIE_RECOMMENDER', 'EXE_TIME', 'Execution time of update_model() for {}_{} is {}.'.format(model_name, model_version, end_time-start_time))
	return jsonify(response), 200
