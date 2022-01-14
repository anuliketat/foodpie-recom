from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, Welcome to Foodpie! <h1>YO!<h1>'

@app.route('/<uid>')
def user(uid):
    return f'Hello {uid}!'

if __name__=='__main__':
    app.run()