from flask import Flask, render_template, request, url_for, flash, redirect

app = Flask(__name__)
#app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506'

@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        arr=[]
        for i in request.form:
            val=request.form[i]
            if val=='':
                return redirect(url_for("signin.html"))
    #return render_template("signin.html")
     

if __name__=="__main__":
    app.run(debug=True)
