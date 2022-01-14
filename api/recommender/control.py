import datetime
import importlib
# import inspect

import requests
from flask import current_app as app

from api.exceptions import NoClass
from utils.misc import logger


class REControl(object):
	def __init__(self, db_main, db_ai, fs_ai):
		self.db_main = db_main
		self.db_ai = db_ai
		self.fs_ai = fs_ai
		self.model_class = None

	def __load_model_class__(self, model_name, model_version):
		try:
			# module_name = 'ml' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
			# _ml_module = importlib.import_module(module_name)

			# Load the module - https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/
			module_name = 'api.recommender.algorithms' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
			module_spec = importlib.util.find_spec(module_name)
			_ml_module = importlib.util.module_from_spec(module_spec)
			module_spec.loader.exec_module(_ml_module)

			# for name, obj in inspect.getmembers(_ml_module):
			# 	print(name)
			# 	if inspect.isclass(obj):
			# 		print(obj)

			try:
				class_name = model_name + '_' + model_version.replace('.', '_')
				_ml_class = getattr(_ml_module, class_name)
				self.model_class = _ml_class()
			except AttributeError as e:
				raise NoClass(e, class_name)
		except Exception as e:
			raise

	def get_food_recommendations(self, model_name, model_version, user_id, N=10):
		logger('FOODPIE_RECOMMENDER', 'REQ', 'get_food_recommendations() of {}_{} called for user_id={} with N={}.'.format(model_name, model_version, user_id, N))
		try:
			self.__load_model_class__(model_name, model_version)
			reco = self.model_class.get_food_recommendations(user_id, N, self.db_main, self.db_ai, self.fs_ai)
		except Exception:
			raise

		food_recommendations = reco['foodRecommendations']
		num_recommendations = len(food_recommendations)
		model_created_at = reco['modelCreatedAt']
		if N != -1 and N <= num_recommendations:
			food_recommendations = food_recommendations[0:N]

		logger('FOODPIE_RECOMMENDER', 'EXE', 'Fetching food recommendations from model {}_{} for user_id={} with N={} successful!'.format(model_name, model_version, user_id, N))
		return {'data': {'customer':user_id, 'foodRecommendations': food_recommendations, 'modelCreatedAt': model_created_at}}

	def update_model(self, model_name, model_version):
		# Use celery or gevent
		logger('FOODPIE_RECOMMENDER', 'REQ', 'update_model() called for: {}_{}.'.format(model_name, model_version))
		self.__load_model_class__(model_name, model_version)
		self.model_class.update_model(self.db_main, self.db_ai, self.fs_ai)
		logger('FOODPIE_RECOMMENDER', 'EXE', 'Update of the model: {}_{} successful!'.format(model_name, model_version))
		return {'message': 'Model has been updated successfully!'}