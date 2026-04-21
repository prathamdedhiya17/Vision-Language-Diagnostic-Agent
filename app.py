from flask import Flask, render_template
from routes.analyze    import analyze_bp
from routes.health     import health_bp
from routes.pdf_report import pdf_bp

app = Flask(__name__)
app.register_blueprint(analyze_bp)
app.register_blueprint(health_bp)
app.register_blueprint(pdf_bp)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
