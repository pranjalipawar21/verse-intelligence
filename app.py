"""
app.py — Verse Intelligence v4
Flask app with: Swagger docs, SQLite history, Celery async, legacy frontend route
"""

import os, logging, traceback
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_restx import Api, Resource, fields

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///translations.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "vi-dev-key-change-in-prod")

db = SQLAlchemy(app)

# ── Swagger ───────────────────────────────────────────────────────────────────
api = Api(app,
    version="4.0",
    title="Verse Intelligence API",
    description="Context-Aware Neural Poetry Translation — EN/HI/MR + 10 European pairs",
    doc="/api/docs",
    prefix="/api",
)

ns_tr  = api.namespace("translate", description="Translation")
ns_his = api.namespace("history",   description="Translation history & stats")
ns_sys = api.namespace("system",    description="Health & languages")

# ── Swagger models ────────────────────────────────────────────────────────────
m_translate = api.model("TranslateRequest", {
    "poem":        fields.String(required=True,  example="Two roads diverged in a yellow wood"),
    "src_lang":    fields.String(required=True,  example="en", description="en hi mr fr de es it ro zh"),
    "tgt_lang":    fields.String(required=True,  example="hi"),
    "mode":        fields.String(example="literal", description="literal | context_aware"),
    "llm_api_key": fields.String(example="sk-...", description="OpenAI or Anthropic key"),
    "use_rag":     fields.Boolean(example=False),
})

# ── DB Model ──────────────────────────────────────────────────────────────────
class Translation(db.Model):
    __tablename__ = "translations"
    id              = db.Column(db.Integer, primary_key=True)
    original        = db.Column(db.Text)
    raw_translation = db.Column(db.Text)
    enhanced        = db.Column(db.Text)
    src_lang        = db.Column(db.String(10))
    tgt_lang        = db.Column(db.String(10))
    mode            = db.Column(db.String(20))
    engine          = db.Column(db.String(20))
    cosine_score    = db.Column(db.Float)
    bleu_score      = db.Column(db.Float)
    bert_f1         = db.Column(db.Float)
    semantic_drift  = db.Column(db.Float)
    sentiment_orig  = db.Column(db.String(20))
    sentiment_trans = db.Column(db.String(20))
    sentiment_drift = db.Column(db.Float)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self, full=False):
        d = dict(id=self.id, src_lang=self.src_lang, tgt_lang=self.tgt_lang,
                 mode=self.mode, engine=self.engine,
                 cosine_score=self.cosine_score, bleu_score=self.bleu_score,
                 bert_f1=self.bert_f1, created_at=self.created_at.isoformat(),
                 original_preview=(self.original or "")[:80]+"…",
                 translated_preview=(self.enhanced or "")[:80]+"…")
        if full:
            d.update(original=self.original, raw_translation=self.raw_translation,
                     enhanced=self.enhanced, semantic_drift=self.semantic_drift,
                     sentiment_orig=self.sentiment_orig, sentiment_trans=self.sentiment_trans,
                     sentiment_drift=self.sentiment_drift)
        return d

with app.app_context():
    db.create_all()

# ── Helpers ───────────────────────────────────────────────────────────────────
def engine_for(src, tgt):
    if "mr" in (src, tgt): return "NLLB-200"
    if "hi" in (src, tgt): return "mBART-50"
    return "MarianMT"

_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline.orchestrator import PoetryPipeline
        _pipeline = PoetryPipeline()
    return _pipeline

def _save(poem, result, src, tgt, mode):
    try:
        m = result.get("metrics", {})
        s = result.get("sentiment", {})
        db.session.add(Translation(
            original=poem, raw_translation=result.get("raw_translation"),
            enhanced=result.get("enhanced_translation"),
            src_lang=src, tgt_lang=tgt, mode=mode, engine=engine_for(src,tgt),
            cosine_score=m.get("cosine_similarity"), bleu_score=m.get("bleu_score"),
            bert_f1=(m.get("bert_score") or {}).get("f1"),
            semantic_drift=m.get("semantic_drift"),
            sentiment_orig=s.get("original",{}).get("label"),
            sentiment_trans=s.get("translated",{}).get("label"),
            sentiment_drift=s.get("drift",{}).get("compound_drift"),
        ))
        db.session.commit()
    except Exception as e:
        logger.warning(f"DB save failed: {e}")
        db.session.rollback()

# ── Frontend route ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Legacy endpoint (used by frontend JS) ────────────────────────────────────
@app.route("/api/translate", methods=["POST"])
def translate_compat():
    data = request.get_json()
    poem = data.get("poem","").strip()
    src  = data.get("src_lang","en")
    tgt  = data.get("tgt_lang","fr")
    if not poem:
        return jsonify({"error":"poem required"}),400
    try:
        result = get_pipeline().run(poem=poem, src_lang=src, tgt_lang=tgt,
            mode=data.get("mode","literal"),
            llm_api_key=data.get("llm_api_key",""),
            use_rag=data.get("use_rag",False))
        _save(poem, result, src, tgt, data.get("mode","literal"))
        return jsonify(result)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error":str(e)}),500

# ── Swagger: Sync translate ───────────────────────────────────────────────────
@ns_tr.route("/sync")
class SyncTranslate(Resource):
    @ns_tr.expect(m_translate)
    @ns_tr.doc(description="Synchronous translation. Recommended for European language pairs.")
    def post(self):
        """Translate a poem (synchronous — waits for result)."""
        data = request.get_json()
        poem = data.get("poem","").strip()
        src  = data.get("src_lang","en")
        tgt  = data.get("tgt_lang","fr")
        if not poem: api.abort(400,"poem required")
        try:
            result = get_pipeline().run(poem=poem, src_lang=src, tgt_lang=tgt,
                mode=data.get("mode","literal"),
                llm_api_key=data.get("llm_api_key",""),
                use_rag=data.get("use_rag",False))
            _save(poem, result, src, tgt, data.get("mode","literal"))
            return result, 200
        except Exception as e:
            api.abort(500, str(e))

# ── Swagger: Async translate ──────────────────────────────────────────────────
@ns_tr.route("/async")
class AsyncTranslate(Resource):
    @ns_tr.expect(m_translate)
    @ns_tr.doc(description="Start translation as background task. Returns task_id. Poll /api/translate/status/<task_id>")
    def post(self):
        """Start async translation (returns task_id immediately)."""
        data = request.get_json()
        poem = data.get("poem","").strip()
        if not poem: api.abort(400,"poem required")
        try:
            from tasks import translate_task
            task = translate_task.delay(poem,
                data.get("src_lang","en"), data.get("tgt_lang","fr"),
                data.get("mode","literal"), data.get("llm_api_key",""),
                data.get("use_rag",False))
            return {"task_id": task.id, "status": "PENDING"}, 202
        except Exception as e:
            logger.warning(f"Celery not available ({e}), running sync fallback")
            result = get_pipeline().run(poem=poem,
                src_lang=data.get("src_lang","en"), tgt_lang=data.get("tgt_lang","fr"),
                mode=data.get("mode","literal"),
                llm_api_key=data.get("llm_api_key",""),
                use_rag=data.get("use_rag",False))
            _save(poem, result, data.get("src_lang","en"), data.get("tgt_lang","fr"), data.get("mode","literal"))
            return {"task_id": None, "status": "SUCCESS", "result": result}, 200

@ns_tr.route("/status/<string:task_id>")
class TaskStatus(Resource):
    @ns_tr.doc(description="Poll background task progress.")
    def get(self, task_id):
        """Get async task status."""
        try:
            from celery.result import AsyncResult
            from tasks import celery
            t = AsyncResult(task_id, app=celery)
            if t.state == "PENDING":
                return {"task_id":task_id,"status":"PENDING","percent":0,"stage":"Queued"}
            if t.state == "PROGRESS":
                meta = t.info or {}
                return {"task_id":task_id,"status":"PROGRESS",
                        "percent":meta.get("percent",0),"stage":meta.get("stage","")}
            if t.state == "SUCCESS":
                return {"task_id":task_id,"status":"SUCCESS","percent":100,"result":t.result}
            return {"task_id":task_id,"status":"FAILURE","error":str(t.info)}
        except Exception as e:
            return {"task_id":task_id,"status":"FAILURE","error":str(e)}

# ── Swagger: History ──────────────────────────────────────────────────────────
@ns_his.route("/")
class HistoryList(Resource):
    def get(self):
        """Get recent translations (paginated)."""
        page = request.args.get("page",1,type=int)
        per  = request.args.get("limit",20,type=int)
        p    = Translation.query.order_by(Translation.created_at.desc()).paginate(page=page,per_page=per,error_out=False)
        return {"total":p.total,"page":p.page,"per_page":p.per_page,
                "translations":[r.to_dict() for r in p.items]}, 200

@ns_his.route("/<int:record_id>")
class HistoryDetail(Resource):
    def get(self, record_id):
        """Get full details of one translation."""
        return Translation.query.get_or_404(record_id).to_dict(full=True), 200

@ns_his.route("/stats")
class Stats(Resource):
    def get(self):
        """Aggregate stats across all saved translations."""
        from sqlalchemy import func
        total = Translation.query.count()
        if not total: return {"total":0}
        avg = db.session.query(
            func.avg(Translation.cosine_score),
            func.avg(Translation.bleu_score),
            func.avg(Translation.bert_f1)).one()
        pairs = (db.session.query(Translation.src_lang, Translation.tgt_lang, func.count())
                 .group_by(Translation.src_lang, Translation.tgt_lang)
                 .order_by(func.count().desc()).limit(5).all())
        return {"total":total,
                "avg_cosine":round(avg[0] or 0,4),
                "avg_bleu":round(avg[1] or 0,4),
                "avg_bert_f1":round(avg[2] or 0,4),
                "top_pairs":[{"src":p[0],"tgt":p[1],"count":p[2]} for p in pairs]}, 200

# ── Swagger: System ───────────────────────────────────────────────────────────
@ns_sys.route("/health")
class Health(Resource):
    def get(self):
        """Health check."""
        return {"status":"ok","service":"Verse Intelligence","version":"4.0"}, 200

@ns_sys.route("/languages")
class Languages(Resource):
    def get(self):
        """All supported language pairs."""
        return [
            {"code":"en-hi","label":"English → Hindi","engine":"mBART-50","size_mb":2300},
            {"code":"en-mr","label":"English → Marathi","engine":"NLLB-200","size_mb":2400},
            {"code":"hi-en","label":"Hindi → English","engine":"mBART-50","size_mb":2300},
            {"code":"mr-en","label":"Marathi → English","engine":"NLLB-200","size_mb":2400},
            {"code":"hi-mr","label":"Hindi → Marathi","engine":"NLLB-200","size_mb":2400},
            {"code":"mr-hi","label":"Marathi → Hindi","engine":"NLLB-200","size_mb":2400},
            {"code":"en-fr","label":"English → French","engine":"MarianMT","size_mb":300},
            {"code":"en-de","label":"English → German","engine":"MarianMT","size_mb":300},
            {"code":"en-es","label":"English → Spanish","engine":"MarianMT","size_mb":300},
            {"code":"en-it","label":"English → Italian","engine":"MarianMT","size_mb":300},
            {"code":"en-ro","label":"English → Romanian","engine":"MarianMT","size_mb":300},
            {"code":"en-zh","label":"English → Chinese","engine":"MarianMT","size_mb":300},
            {"code":"fr-en","label":"French → English","engine":"MarianMT","size_mb":300},
            {"code":"de-en","label":"German → English","engine":"MarianMT","size_mb":300},
            {"code":"es-en","label":"Spanish → English","engine":"MarianMT","size_mb":300},
            {"code":"it-en","label":"Italian → English","engine":"MarianMT","size_mb":300},
        ], 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
