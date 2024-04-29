from re import X
from unicodedata import category
from flask import Blueprint, render_template, request, flash
from prediction import manual_testing
views = Blueprint("views", __name__)

@views.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = request.form.get("add")
        print(data)
        output = (manual_testing(data))

        if output == True:
            flash("This source is secure and should be safe to use ", category="success")
        elif output == False:
            flash("This source is fake and unreliable. Please use this source with caution.", category="error")
        
    return render_template("home.html")