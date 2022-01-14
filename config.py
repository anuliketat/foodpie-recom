import os

# Set the environment whether devopment, test or production
# and configure according to that environment
DEBUG = True if os.environ.get('DEBUG') == 'True' else False
TESTING = True if os.environ.get('TESTING') == 'True' else False

ECOSYSTEM = os.environ.get('ECOSYSTEM')
if ECOSYSTEM is None:
    if DEBUG is True:
        ECOSYSTEM = 'dev'
    elif TESTING is True:
        ECOSYSTEM = 'test'
    else:
        ECOSYSTEM = 'prod'
if ECOSYSTEM not in ['dev', 'test', 'prod']:
    raise ValueError("ECOSYSTEM has to be 'dev', 'test' or 'prod'")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# MongoDB conenction parameters
MONGO_HOST = os.environ.get('MONGO_HOST', '127.0.0.1')
MONGO_PORT = os.environ.get('MONGO_PORT', '27017')
MONGO_DB_MAIN = os.environ.get('MONGO_DB_MAIN', 'recom')
MONGO_DB_AI = os.environ.get('MONGO_DB_AI', 'foodpie_ai')

if os.environ.get('MONGO_AUTH') == 'True':
    MONGO_AUTH = True
else:
    MONGO_AUTH = False
if MONGO_AUTH:
    MONGO_USERNAME = os.environ.get('MONGO_USERNAME', 'foodpie_ai')
    MONGO_PASSWORD = os.environ.get('MONGO_PASSWORD')
    MONGO_AUTH_SOURCE = os.environ.get('MONGO_AUTH_SOURCE', 'admin')

MONGO_URI_MAIN = (
             "mongodb://"
             + MONGO_HOST + ":"
             + MONGO_PORT + "/"
             + MONGO_DB_MAIN
            )
MONGO_URI_AI = (
             "mongodb://"
             + MONGO_HOST + ":"
             + MONGO_PORT + "/"
             + MONGO_DB_AI
            )