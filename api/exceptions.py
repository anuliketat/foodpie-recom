from utils.misc import get_traceback


class NoClass(Exception):
	def __init__(self, e, class_name):
		self.e = e
		self.class_name = class_name

	def http_status(self):
		return 400

	def traceback(self):
		return get_traceback(self.e)

	def __str__(self):
		return 'ERROR! Class: {} does not exist! Please check the model and version!'.format(self.class_name)

class NoModel(Exception):
	def __init__(self, model_name, model_version):
		self.model_name = model_name
		self.model_version = model_version

	def http_status(self):
		return 426

	def __str__(self):
		return 'ALERT! The ML Model: {}_{} is not yet trained! Please update the model!'.format(self.model_name, self.model_version)


class DBError(Exception):
	def __init__(self, e, db_name, error):
		self.e = e
		self.db_name = db_name
		self.error = error

	def http_status(self):
		return 500

	def traceback(self):
		return get_traceback(self.e)

	def __str__(self):
		return 'ERROR! DB name: {} and error message: {}!'.format(self.db_name, self.error)
