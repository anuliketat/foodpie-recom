import os
import sys

from api import app

# import nose2

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
