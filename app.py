from flask import Flask, request, render_template
from movie_recommendation import *

# create the flask app
app = Flask(__name__)

# @app.route is used to map the specific URL with the associated function that is intended to perform some task
@app.route("/")
def home():
    return render_template('home.html')

@app.route('/', methods=['GET','POST'])
def recommend_movies():
    if request.method == 'POST':
        # get the movie_name from the input in html file associated to the 'movie' name
        movie_name = request.form['movie']
        result = results(movie_name)
        # if the movie is not present in the dataset I display the notFound html page
        # that indicates to user to enter another movie
        if result == 'Movie not in Database':
            return render_template('notFound.html', name=movie_name)
        else:
            recommendations = get_name(result)
    return render_template('show.html', movies = recommendations)

if __name__ == '__main__':
    app.run(debug=True)
