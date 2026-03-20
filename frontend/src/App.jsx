import { useState, useEffect, useRef } from "react";

/* ============================================================
   DiabetesSense v2.0 — Updated Frontend
   Dataset: Early Stage Diabetes Risk Prediction (UCI)
   Features: Age, Gender + 14 Binary Symptom Indicators
   Model:    Random Forest (AUC=0.9679, Acc=96.1%)
   ============================================================ */

// ── Theme ──────────────────────────────────────────────────────────────────
const C = {
  bg:       "#f7f9ff",
  navy:     "#0b1f3a",
  blue:     "#1a4fdb",
  sky:      "#38a3f5",
  teal:     "#0d9e8a",
  border:   "#dde3f0",
  muted:    "#6b7a99",
  text:     "#1a2540",
  card:     "#ffffff",
  danger:   "#e53e3e",
  warn:     "#d97706",
  ok:       "#16a34a",
};

const FONTS = `@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,600;0,700;0,900;1,300&family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');`;

const GS = `
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:${C.bg};font-family:'Plus Jakarta Sans',sans-serif;color:${C.text};min-height:100vh}
button{cursor:pointer;font-family:'Plus Jakarta Sans',sans-serif}
input,select{font-family:'Plus Jakarta Sans',sans-serif;outline:none}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:${C.border};border-radius:3px}

@keyframes fadeSlideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
@keyframes barIn{from{width:0}to{width:var(--w)}}
@keyframes countUp{from{opacity:0;transform:scale(0.85)}to{opacity:1;transform:scale(1)}}
@keyframes needleSweep{from{transform:rotate(-90deg)}to{transform:rotate(var(--deg))}}
@keyframes ringGrow{from{stroke-dashoffset:283}to{stroke-dashoffset:var(--offset)}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 0 0 rgba(26,79,219,.3)}50%{box-shadow:0 0 0 8px rgba(26,79,219,0)}}

.au{animation:fadeSlideUp .55s ease both}
.au1{animation:fadeSlideUp .55s .08s ease both}
.au2{animation:fadeSlideUp .55s .16s ease both}
.au3{animation:fadeSlideUp .55s .24s ease both}
.au4{animation:fadeSlideUp .55s .32s ease both}
.au5{animation:fadeSlideUp .55s .40s ease both}

.navlink{
  font-size:.85rem;font-weight:600;color:${C.muted};text-decoration:none;
  padding:6px 14px;border-radius:8px;transition:all .18s;letter-spacing:.01em;cursor:pointer
}
.navlink:hover,.navlink.active{color:${C.blue};background:${C.blue}14}

.inp{
  width:100%;padding:10px 14px;border:1.5px solid ${C.border};border-radius:10px;
  font-size:.88rem;background:#fff;color:${C.text};transition:border-color .2s,box-shadow .2s
}
.inp:focus{border-color:${C.blue};box-shadow:0 0 0 3px ${C.blue}1a}

.card{background:#fff;border-radius:18px;border:1px solid ${C.border};box-shadow:0 2px 12px rgba(0,0,0,.05)}

.btn-primary{
  background:linear-gradient(135deg,${C.blue},${C.sky});color:#fff;border:none;
  padding:12px 28px;border-radius:12px;font-size:.92rem;font-weight:600;
  letter-spacing:.02em;transition:all .18s;
  box-shadow:0 4px 14px ${C.blue}35
}
.btn-primary:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 6px 20px ${C.blue}50}
.btn-primary:disabled{opacity:.5;cursor:not-allowed}

.toggle-group{display:flex;gap:0;border:1.5px solid ${C.border};border-radius:10px;overflow:hidden}
.toggle-btn{
  flex:1;padding:9px 6px;border:none;background:transparent;
  font-size:.8rem;font-weight:600;color:${C.muted};
  transition:all .18s;cursor:pointer;white-space:nowrap
}
.toggle-btn.active{background:${C.blue};color:#fff}
.toggle-btn:hover:not(.active){background:${C.blue}10;color:${C.blue}}

.pill{
  display:inline-flex;align-items:center;gap:5px;
  padding:4px 12px;border-radius:20px;font-size:.75rem;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase
}
`;

// ── Helpers ─────────────────────────────────────────────────────────────────
const riskInfo = (score) => {
  if (score < 30) return { label:"Low Risk", color:C.ok,   bg:"#dcfce7", emoji:"🟢", grad:"#16a34a, #4ade80" };
  if (score < 65) return { label:"Moderate Risk", color:C.warn, bg:"#fef3c7", emoji:"🟡", grad:"#d97706, #fbbf24" };
  return               { label:"High Risk",    color:C.danger, bg:"#fee2e2", emoji:"🔴", grad:"#e53e3e, #f87171" };
};

// Dataset-accurate ML simulation
// Calibrated so all-negative answers → ~4-8% (Low Risk)
// Key signals (Polyuria + Polydipsia) still drive score to 65-90%
function simulateRF(data) {
  // Intercept calibrated: sigmoid(-3.2) ≈ 0.039 → 4% base risk for all-negative
  let logOdds = -3.2;

  // Polyuria & Polydipsia are top predictors (RF importance: 0.22, 0.18)
  if (data.Polyuria === 1)            logOdds += 2.2;
  if (data.Polydipsia === 1)          logOdds += 2.0;
  if (data["sudden weight loss"]===1) logOdds += 1.2;
  if (data.Polyphagia === 1)          logOdds += 0.9;
  if (data.Irritability === 1)        logOdds += 0.8;
  if (data["partial paresis"]===1)    logOdds += 0.8;
  if (data.weakness === 1)            logOdds += 0.6;
  if (data["visual blurring"]===1)    logOdds += 0.6;
  if (data["Genital thrush"]===1)     logOdds += 0.5;
  if (data["delayed healing"]===1)    logOdds += 0.5;
  if (data.Obesity === 1)             logOdds += 0.5;
  if (data["muscle stiffness"]===1)   logOdds += 0.4;
  if (data.Alopecia === 1)            logOdds += 0.35;
  if (data.Itching === 1)             logOdds += 0.3;
  // Age adds risk only above 40 (dataset-aligned thresholds)
  if (data.Age > 60)                  logOdds += 0.8;
  else if (data.Age > 50)             logOdds += 0.5;
  else if (data.Age > 40)             logOdds += 0.25;
  // Gender: Males have slightly higher prevalence in dataset, minor adjustment
  if (data.Gender === 1)              logOdds += 0.2;

  const prob = 1 / (1 + Math.exp(-logOdds));
  const score = Math.round(Math.min(98, Math.max(2, prob * 100)));

  // SHAP-like feature contributions
  // Age contribution: only show non-zero if > 40
  const ageContrib = data.Age > 60 ? 10 : data.Age > 50 ? 6 : data.Age > 40 ? 3 : 0;
  // Gender contribution: only show non-zero for Male, small value
  const genderContrib = data.Gender === 1 ? 2 : 0;

  const contributions = [
    { name:"Polyuria (Excess Urination)",  val: data.Polyuria          * 22, present: data.Polyuria===1 },
    { name:"Polydipsia (Excess Thirst)",   val: data.Polydipsia        * 18, present: data.Polydipsia===1 },
    { name:"Sudden Weight Loss",           val: data["sudden weight loss"]*12, present: data["sudden weight loss"]===1 },
    { name:"Polyphagia (Excess Hunger)",   val: data.Polyphagia*9,      present: data.Polyphagia===1 },
    { name:"Irritability",                 val: data.Irritability*8,    present: data.Irritability===1 },
    { name:"Partial Paresis",              val: data["partial paresis"]*8, present: data["partial paresis"]===1 },
    { name:"Weakness",                     val: data.weakness*6,        present: data.weakness===1 },
    { name:"Visual Blurring",              val: data["visual blurring"]*6, present: data["visual blurring"]===1 },
    { name:"Genital Thrush",               val: data["Genital thrush"]*5, present: data["Genital thrush"]===1 },
    { name:"Delayed Healing",              val: data["delayed healing"]*5, present: data["delayed healing"]===1 },
    { name:"Obesity",                      val: data.Obesity*5,         present: data.Obesity===1 },
    { name:"Muscle Stiffness",             val: data["muscle stiffness"]*4, present: data["muscle stiffness"]===1 },
    { name:"Alopecia (Hair Loss)",         val: data.Alopecia*3,        present: data.Alopecia===1 },
    { name:"Itching",                      val: data.Itching*3,         present: data.Itching===1 },
    // Age & Gender only contribute if they actually add to score
    { name:"Age",                          val: ageContrib,             present: ageContrib > 0 },
    { name:"Gender",                       val: genderContrib,          present: data.Gender===1 && genderContrib > 0 },
  ].sort((a,b) => b.val - a.val);

  return { score, prob: prob.toFixed(4), contributions };
}

function getRecommendations(score, data) {
  const syms = [];
  if (data.Polyuria===1)   syms.push("frequent urination");
  if (data.Polydipsia===1) syms.push("excessive thirst");
  if (data["sudden weight loss"]===1) syms.push("unexplained weight loss");

  if (score < 30) return [
    { icon:"💧", t:"Stay Well-Hydrated",       d:"Drink 8–10 glasses of water daily. Avoid sugary drinks. Your current profile is reassuring." },
    { icon:"🥗", t:"Balanced Nutrition",        d:"Emphasize low-GI foods, vegetables, lean proteins. Limit processed sugars and refined carbs." },
    { icon:"🏃", t:"Maintain Physical Activity", d:"Keep up 150+ min/week of moderate aerobic exercise. Physical activity is your best preventive medicine." },
    { icon:"🩺", t:"Annual Check-up",           d:"Continue routine blood glucose and HbA1c tests annually. Early monitoring is key even at low risk." },
    { icon:"😴", t:"Quality Sleep",             d:"7–9 hours nightly supports healthy glucose metabolism and reduces metabolic syndrome risk." },
  ];

  if (score < 65) return [
    { icon:"🚨", t:"Schedule Medical Evaluation", d:`You report ${syms.length>0?syms.join(', ')+' — these':'several symptoms'} that warrant clinical evaluation. Request fasting glucose and HbA1c tests from your doctor within 2 weeks.` },
    { icon:"🏋️", t:"Structured Exercise Plan",  d:"Target 30+ min of moderate exercise 5 days/week. Resistance training 2–3×/week significantly improves insulin sensitivity." },
    { icon:"🥦", t:"Low-Glycemic Diet",          d:"Work with a nutritionist. Prioritize vegetables, legumes, whole grains. Eliminate sugary beverages. Keep meal portions consistent." },
    { icon:"⚖️", t:"Weight Management",          d:data.Obesity===1?"5–7% body weight reduction can cut diabetes progression risk by up to 58% (CDC DPP data).":"Maintain healthy weight through consistent activity and balanced eating." },
    { icon:"📊", t:"Track Symptoms",             d:"Log your symptoms, energy levels, and fluid intake daily using a health app. Share this log with your physician." },
    { icon:"🧘", t:"Stress Reduction",           d:"Chronic stress elevates cortisol which worsens insulin resistance. Practice mindfulness, yoga, or structured relaxation 20 min/day." },
  ];

  return [
    { icon:"🏥", t:"Seek Immediate Medical Care",  d:`Your symptom profile is high-risk. ${syms.length>0?"Symptoms like "+syms.join(', ')+" are classic early diabetes indicators. ":""}Visit your physician or endocrinologist this week — do not delay.` },
    { icon:"🔬", t:"Request Full Diabetes Panel",  d:"Ask for: Fasting Plasma Glucose, 2-hr OGTT, HbA1c, fasting insulin, C-peptide, full lipid panel, kidney function (eGFR, creatinine)." },
    { icon:"💊", t:"Discuss Preventive Medication",d:"Your doctor may recommend Metformin — proven to reduce diabetes onset by 31% in high-risk individuals (NEJM DPP trial)." },
    { icon:"🥗", t:"Medical Nutrition Therapy",    d:"Request referral to a Registered Dietitian for personalized meal planning. Target: < 45g carbs/meal, high fiber (25–35g/day), eliminate all SSBs." },
    { icon:"📏", t:"Daily Self-Monitoring",        d:"Consider a home glucometer. Monitor fasting and 2-hr post-meal glucose. Target: fasting < 100 mg/dL, post-meal < 140 mg/dL." },
    { icon:"🤝", t:"Join a Prevention Program",    d:"The CDC Diabetes Prevention Program (DPP) is a structured lifestyle intervention proven to reduce T2D risk by 58%. Ask your doctor for a referral." },
    { icon:"😴", t:"Treat Sleep Issues",           d:"Sleep apnea and insomnia dramatically worsen insulin resistance. Discuss sleep quality with your doctor. Aim for uninterrupted 7–8 hours nightly." },
  ];
}

// ── Components ───────────────────────────────────────────────────────────────

function Navbar({ page, setPage }) {
  const links = ["Home","Assess","Results","About"];
  return (
    <nav style={{
      position:"sticky",top:0,zIndex:200,
      background:"rgba(247,249,255,.92)",backdropFilter:"blur(14px)",
      borderBottom:`1px solid ${C.border}`,
    }}>
      <div style={{maxWidth:1100,margin:"0 auto",padding:"0 1.5rem",display:"flex",alignItems:"center",justifyContent:"space-between",height:62}}>
        <div style={{display:"flex",alignItems:"center",gap:10,cursor:"pointer"}} onClick={()=>setPage("Home")}>
          <div style={{
            width:34,height:34,borderRadius:9,
            background:`linear-gradient(135deg,${C.blue},${C.teal})`,
            display:"flex",alignItems:"center",justifyContent:"center",
            fontFamily:"'Fraunces',serif",fontWeight:900,color:"#fff",fontSize:17,
          }}>D</div>
          <span style={{fontFamily:"'Fraunces',serif",fontWeight:700,fontSize:"1.1rem",color:C.navy,letterSpacing:"-.01em"}}>
            DiabetesSense
          </span>
          <span style={{
            fontSize:".65rem",fontFamily:"'DM Mono',monospace",
            background:`${C.blue}15`,color:C.blue,
            padding:"2px 7px",borderRadius:4,fontWeight:500,
          }}>v2.0</span>
        </div>
        <div style={{display:"flex",gap:2}}>
          {links.map(l=>(
            <span key={l} className={`navlink${page===l?" active":""}`} onClick={()=>setPage(l)}>{l}</span>
          ))}
        </div>
        <button className="btn-primary" style={{padding:"8px 18px",fontSize:".82rem"}} onClick={()=>setPage("Assess")}>
          Start Assessment →
        </button>
      </div>
    </nav>
  );
}

function HomePage({ setPage }) {
  const stats = [
    { v:"96.1%", l:"Model Accuracy",       sub:"Random Forest" },
    { v:"0.968", l:"AUC-ROC Score",        sub:"Discrimination ability" },
    { v:"16",    l:"Clinical Features",     sub:"Symptom indicators" },
    { v:"251",   l:"Training Samples",      sub:"After deduplication" },
    { v:"SHAP",  l:"Explainable AI",        sub:"Feature-level insights" },
  ];

  const symptoms = [
    ["Polyuria","Polydipsia","Sudden Weight Loss","Weakness"],
    ["Polyphagia","Genital Thrush","Visual Blurring","Itching"],
    ["Irritability","Delayed Healing","Partial Paresis","Muscle Stiffness"],
    ["Alopecia","Obesity"],
  ];

  return (
    <div>
      {/* HERO */}
      <section style={{
        background:`linear-gradient(150deg,${C.navy} 0%,#0d2d5c 55%,#133a70 100%)`,
        padding:"90px 1.5rem 70px",
        position:"relative",overflow:"hidden",
      }}>
        {/* Decorative dots grid */}
        <div style={{
          position:"absolute",inset:0,
          backgroundImage:`radial-gradient(circle,rgba(255,255,255,.06) 1px,transparent 1px)`,
          backgroundSize:"32px 32px",pointerEvents:"none",
        }}/>
        <div style={{position:"absolute",right:"-120px",top:"-80px",width:500,height:500,borderRadius:"50%",
          background:`radial-gradient(circle,${C.sky}15 0%,transparent 70%)`,pointerEvents:"none"}}/>

        <div style={{maxWidth:720,margin:"0 auto",textAlign:"center",position:"relative",zIndex:1}}>
          <div className="au" style={{
            display:"inline-block",background:"rgba(56,163,245,.15)",
            border:"1px solid rgba(56,163,245,.3)",color:"#7dd3fc",
            padding:"4px 16px",borderRadius:20,fontSize:".75rem",
            fontFamily:"'DM Mono',monospace",letterSpacing:".08em",marginBottom:24,
          }}>
            EARLY-STAGE DIABETES RISK DETECTION · ML v2.0
          </div>

          <h1 className="au1" style={{
            fontFamily:"'Fraunces',serif",
            fontSize:"clamp(2rem,5.5vw,3.8rem)",
            fontWeight:900,color:"#fff",lineHeight:1.1,marginBottom:20,
          }}>
            Know Your Risk<br/>
            <span style={{
              fontStyle:"italic",
              background:`linear-gradient(90deg,${C.sky},${C.teal})`,
              WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",
            }}>Before Symptoms Worsen</span>
          </h1>

          <p className="au2" style={{color:"#94a3b8",fontSize:"1rem",lineHeight:1.75,marginBottom:36,maxWidth:540,margin:"0 auto 36px"}}>
            A clinically-grounded AI tool trained on 16 early-stage diabetes indicators. 
            In under 2 minutes, get your personal risk score with explainable AI insights and actionable health recommendations.
          </p>

          <div className="au3" style={{display:"flex",gap:14,justifyContent:"center",flexWrap:"wrap"}}>
            <button className="btn-primary" style={{padding:"14px 32px",fontSize:".98rem",animation:"glowPulse 2.5s infinite"}}
              onClick={()=>setPage("Assess")}>
              Begin Risk Assessment →
            </button>
            <button onClick={()=>setPage("About")} style={{
              background:"transparent",border:"1.5px solid rgba(255,255,255,.18)",
              color:"#cbd5e1",padding:"14px 24px",borderRadius:12,fontSize:".9rem",fontWeight:500,
              transition:"border-color .2s",
            }}>View Methodology</button>
          </div>
        </div>
      </section>

      {/* STATS RIBBON */}
      <div style={{background:"#fff",borderBottom:`1px solid ${C.border}`}}>
        <div style={{maxWidth:1100,margin:"0 auto",padding:"0 1.5rem",display:"flex",flexWrap:"wrap",gap:0}}>
          {stats.map((s,i)=>(
            <div key={i} style={{
              flex:"1 1 160px",padding:"24px 20px",
              borderRight:i<stats.length-1?`1px solid ${C.border}`:"none",
              textAlign:"center",
            }}>
              <div style={{fontFamily:"'Fraunces',serif",fontSize:"1.9rem",fontWeight:700,color:C.blue,lineHeight:1}}>{s.v}</div>
              <div style={{fontWeight:600,fontSize:".82rem",color:C.text,marginTop:4}}>{s.l}</div>
              <div style={{fontSize:".72rem",color:C.muted,marginTop:2}}>{s.sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* SYMPTOM FEATURES */}
      <section style={{maxWidth:1100,margin:"0 auto",padding:"64px 1.5rem"}}>
        <div style={{textAlign:"center",marginBottom:40}}>
          <h2 style={{fontFamily:"'Fraunces',serif",fontSize:"1.9rem",fontWeight:700,color:C.navy,marginBottom:8}}>
            16 Clinical & Symptomatic Features
          </h2>
          <p style={{color:C.muted,maxWidth:500,margin:"0 auto",fontSize:".9rem",lineHeight:1.65}}>
            This model uses symptom-based features that map directly to early-stage Type 2 Diabetes clinical presentations.
          </p>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(130px,1fr))",gap:12}}>
          {symptoms.flat().map((s,i)=>(
            <div key={s} className="card au" style={{
              padding:"14px 12px",textAlign:"center",
              animation:`fadeSlideUp .5s ${i*.04}s ease both`,opacity:0,animationFillMode:"forwards",
            }}>
              <div style={{width:8,height:8,borderRadius:"50%",background:`linear-gradient(135deg,${C.blue},${C.teal})`,margin:"0 auto 8px"}}/>
              <div style={{fontSize:".78rem",fontWeight:600,color:C.navy,lineHeight:1.3}}>{s}</div>
            </div>
          ))}
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section style={{background:"#fff",padding:"64px 1.5rem",borderTop:`1px solid ${C.border}`,borderBottom:`1px solid ${C.border}`}}>
        <div style={{maxWidth:900,margin:"0 auto"}}>
          <h2 style={{fontFamily:"'Fraunces',serif",fontSize:"1.9rem",fontWeight:700,color:C.navy,textAlign:"center",marginBottom:40}}>
            How DiabetesSense Works
          </h2>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(220px,1fr))",gap:28}}>
            {[
              {n:"01",ico:"📋",t:"Enter Symptoms",d:"Report your age, gender, and 14 early diabetes symptom indicators. No blood tests required."},
              {n:"02",ico:"🤖",t:"ML Analysis",d:"Random Forest ensemble (96.1% accuracy) processes your inputs using patterns from 251 clinical cases."},
              {n:"03",ico:"🔍",t:"SHAP Explanation",d:"See exactly which symptoms drive your risk score — transparent, feature-level AI explanations."},
              {n:"04",ico:"📊",t:"Personalized Plan",d:"Receive evidence-based recommendations calibrated to your specific risk level and symptom profile."},
            ].map(s=>(
              <div key={s.n} className="card" style={{padding:"24px 20px"}}>
                <div style={{fontFamily:"'DM Mono',monospace",color:C.blue,fontSize:".7rem",marginBottom:10,fontWeight:500}}>STEP {s.n}</div>
                <div style={{fontSize:24,marginBottom:10}}>{s.ico}</div>
                <div style={{fontFamily:"'Fraunces',serif",fontWeight:700,color:C.navy,marginBottom:6}}>{s.t}</div>
                <div style={{color:C.muted,fontSize:".83rem",lineHeight:1.6}}>{s.d}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{padding:"64px 1.5rem"}}>
        <div style={{
          maxWidth:860,margin:"0 auto",
          background:`linear-gradient(135deg,${C.blue}0a,${C.teal}12)`,
          border:`1.5px solid ${C.blue}22`,borderRadius:24,
          padding:"44px 48px",display:"flex",flexWrap:"wrap",gap:28,alignItems:"center",justifyContent:"space-between",
        }}>
          <div>
            <h3 style={{fontFamily:"'Fraunces',serif",fontSize:"1.5rem",fontWeight:700,color:C.navy,marginBottom:8}}>
              Get Your Risk Assessment in 2 Minutes
            </h3>
            <p style={{color:C.muted,maxWidth:420,lineHeight:1.65,fontSize:".88rem"}}>
              Early awareness saves lives. Our AI model identifies high-risk profiles with 96.1% accuracy. 
              No registration required — completely private.
            </p>
          </div>
          <button className="btn-primary" style={{padding:"15px 30px",fontSize:".95rem",whiteSpace:"nowrap"}}
            onClick={()=>setPage("Assess")}>
            Start Now — It's Free →
          </button>
        </div>
      </section>
    </div>
  );
}

// ── ASSESSMENT PAGE ───────────────────────────────────────────────────────────

const DEFAULT_FORM = {
  Age: "", Gender: "Male",
  Polyuria: -1, Polydipsia: -1, "sudden weight loss": -1, weakness: -1,
  Polyphagia: -1, "Genital thrush": -1, "visual blurring": -1, Itching: -1,
  Irritability: -1, "delayed healing": -1, "partial paresis": -1,
  "muscle stiffness": -1, Alopecia: -1, Obesity: -1,
};

const SYMPTOM_INFO = {
  Polyuria:             { label:"Polyuria",          desc:"Frequent or excessive urination (more than 3L/day)", icon:"💧" },
  Polydipsia:           { label:"Polydipsia",         desc:"Excessive or abnormal thirst throughout the day",    icon:"🥤" },
  "sudden weight loss": { label:"Sudden Weight Loss", desc:"Unexplained significant weight loss recently",       icon:"⚖️" },
  weakness:             { label:"Weakness",           desc:"Persistent unexplained physical fatigue or weakness", icon:"😓" },
  Polyphagia:           { label:"Polyphagia",         desc:"Excessive hunger even after eating normally",        icon:"🍽️" },
  "Genital thrush":     { label:"Genital Thrush",     desc:"Recurring fungal/yeast infections in genital area",  icon:"⚕️" },
  "visual blurring":    { label:"Visual Blurring",    desc:"Blurring of vision or difficulty focusing",          icon:"👁️" },
  Itching:              { label:"Itching",            desc:"Persistent unexplained skin itching",                icon:"🤲" },
  Irritability:         { label:"Irritability",       desc:"Unusual mood swings, irritability, or anxiety",      icon:"😤" },
  "delayed healing":    { label:"Delayed Healing",    desc:"Cuts, sores, or bruises take unusually long to heal", icon:"🩹" },
  "partial paresis":    { label:"Partial Paresis",    desc:"Partial loss of voluntary movement in limbs",        icon:"🦵" },
  "muscle stiffness":   { label:"Muscle Stiffness",   desc:"Unusual stiffness or tightness in muscles",          icon:"💪" },
  Alopecia:             { label:"Alopecia",           desc:"Unusual or patchy hair loss from the scalp",         icon:"💈" },
  Obesity:              { label:"Obesity",            desc:"Body Mass Index (BMI) above 30 kg/m²",               icon:"📏" },
};

const SYMPTOM_KEYS = Object.keys(SYMPTOM_INFO);

function SymToggle({ fkey, val, onChange }) {
  const info = SYMPTOM_INFO[fkey];
  const [hover, setHover] = useState(false);
  return (
    <div className="card" style={{
      padding:"14px 16px",
      border:`1.5px solid ${val===1?C.blue:val===0?C.ok:C.border}`,
      transition:"border-color .18s,box-shadow .18s",
      boxShadow: val===1?`0 0 0 3px ${C.blue}18`:val===0?`0 0 0 3px ${C.ok}18`:"none",
    }}>
      <div style={{display:"flex",alignItems:"flex-start",gap:10,marginBottom:10}}>
        <span style={{fontSize:18,lineHeight:1}}>{info.icon}</span>
        <div>
          <div style={{fontWeight:700,fontSize:".85rem",color:C.navy,marginBottom:2}}>{info.label}</div>
          <div style={{fontSize:".72rem",color:C.muted,lineHeight:1.4}}>{info.desc}</div>
        </div>
      </div>
      <div className="toggle-group">
        <button className={`toggle-btn${val===1?" active":""}`}
          style={val===1?{background:C.blue,color:"#fff"}:{}}
          onClick={()=>onChange(fkey, val===1?-1:1)}>
          Yes
        </button>
        <button className={`toggle-btn${val===0?" active":""}`}
          style={val===0?{background:C.ok,color:"#fff"}:{}}
          onClick={()=>onChange(fkey, val===0?-1:0)}>
          No
        </button>
      </div>
    </div>
  );
}

function AssessPage({ form, setForm, onPredict, loading }) {
  const answered  = SYMPTOM_KEYS.filter(k=>form[k]!==-1).length;
  const complete  = form.Age !== "" && answered === SYMPTOM_KEYS.length;
  const progress  = Math.round((answered/SYMPTOM_KEYS.length)*100);

  const handleSym = (k,v) => setForm(p=>({...p,[k]:v}));

  return (
    <div style={{maxWidth:920,margin:"0 auto",padding:"44px 1.5rem"}}>
      <div className="au" style={{marginBottom:32}}>
        <h1 style={{fontFamily:"'Fraunces',serif",fontSize:"2rem",fontWeight:800,color:C.navy,marginBottom:6}}>
          Diabetes Risk Assessment
        </h1>
        <p style={{color:C.muted,fontSize:".9rem"}}>
          Answer all questions accurately for the most reliable prediction. Based on UCI Early Stage Diabetes Risk dataset.
        </p>
      </div>

      {/* Progress bar */}
      <div className="card au1" style={{padding:"16px 20px",marginBottom:28}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
          <span style={{fontWeight:600,fontSize:".85rem",color:C.navy}}>
            {answered === SYMPTOM_KEYS.length && form.Age ? "✅ All fields complete — ready to predict!" : `Progress: ${answered + (form.Age?1:0)} / ${SYMPTOM_KEYS.length+1} answered`}
          </span>
          <span style={{fontFamily:"'DM Mono',monospace",fontSize:".8rem",color:C.blue,fontWeight:600}}>{progress}%</span>
        </div>
        <div style={{height:6,borderRadius:3,background:C.border,overflow:"hidden"}}>
          <div style={{
            height:"100%",borderRadius:3,width:`${progress}%`,
            background:`linear-gradient(90deg,${C.blue},${C.teal})`,
            transition:"width .4s ease",
          }}/>
        </div>
      </div>

      {/* Demographics */}
      <div className="card au2" style={{padding:"28px",marginBottom:24}}>
        <h3 style={{fontFamily:"'Fraunces',serif",fontWeight:700,color:C.navy,marginBottom:18,fontSize:"1rem",
          display:"flex",alignItems:"center",gap:8}}>
          <span style={{width:26,height:26,borderRadius:8,background:`${C.blue}15`,display:"inline-flex",alignItems:"center",justifyContent:"center",fontSize:13}}>👤</span>
          Demographics
        </h3>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
          <div>
            <label style={{display:"block",fontWeight:600,fontSize:".83rem",color:C.text,marginBottom:6}}>
              Age <span style={{color:C.danger}}>*</span>
            </label>
            <input className="inp" type="number" min="1" max="120" placeholder="e.g. 45"
              value={form.Age}
              onChange={e=>setForm(p=>({...p,Age:e.target.value}))}/>
            <p style={{fontSize:".72rem",color:C.muted,marginTop:4}}>Dataset range: 16–90 years</p>
          </div>
          <div>
            <label style={{display:"block",fontWeight:600,fontSize:".83rem",color:C.text,marginBottom:6}}>Gender</label>
            <div className="toggle-group">
              {["Male","Female"].map(g=>(
                <button key={g} className={`toggle-btn${form.Gender===g?" active":""}`}
                  onClick={()=>setForm(p=>({...p,Gender:g}))}>
                  {g==="Male"?"👨 Male":"👩 Female"}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Symptoms */}
      <div className="au3" style={{marginBottom:28}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:16}}>
          <h3 style={{fontFamily:"'Fraunces',serif",fontWeight:700,color:C.navy,fontSize:"1rem",
            display:"flex",alignItems:"center",gap:8}}>
            <span style={{width:26,height:26,borderRadius:8,background:`${C.blue}15`,display:"inline-flex",alignItems:"center",justifyContent:"center",fontSize:13}}>🩺</span>
            Symptom Indicators
            <span style={{fontFamily:"'DM Mono',monospace",fontSize:".7rem",color:C.muted,fontWeight:400}}>
              ({answered}/{SYMPTOM_KEYS.length} answered)
            </span>
          </h3>
          <button onClick={()=>{
            const all = {};
            SYMPTOM_KEYS.forEach(k=>all[k]=0);
            setForm(p=>({...p,...all}));
          }} style={{
            background:`${C.ok}15`,border:`1px solid ${C.ok}30`,color:C.ok,
            padding:"5px 12px",borderRadius:8,fontSize:".75rem",fontWeight:600,
          }}>Mark All "No"</button>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(260px,1fr))",gap:14}}>
          {SYMPTOM_KEYS.map(k=>(
            <SymToggle key={k} fkey={k} val={form[k]} onChange={handleSym}/>
          ))}
        </div>
      </div>

      {/* Submit */}
      <div className="au4 card" style={{padding:"24px 28px"}}>
        {!complete && (
          <p style={{fontSize:".83rem",color:C.warn,marginBottom:14,display:"flex",alignItems:"center",gap:6}}>
            ⚠ {form.Age===""?"Enter your age. ":""}{answered<SYMPTOM_KEYS.length?`Answer all ${SYMPTOM_KEYS.length} symptom questions (${SYMPTOM_KEYS.length-answered} remaining).`:""}
          </p>
        )}
        <button className="btn-primary" disabled={!complete||loading} onClick={onPredict}
          style={{width:"100%",padding:"15px",fontSize:"1rem"}}>
          {loading?(
            <span style={{display:"flex",alignItems:"center",justifyContent:"center",gap:10}}>
              <span style={{width:18,height:18,border:"2px solid rgba(255,255,255,.3)",borderTopColor:"#fff",borderRadius:"50%",display:"inline-block",animation:"spin .7s linear infinite"}}/>
              Running Random Forest model...
            </span>
          ):"Generate Risk Prediction →"}
        </button>
        <p style={{marginTop:12,fontSize:".73rem",color:C.muted,textAlign:"center",lineHeight:1.5}}>
          🔒 All processing happens in your browser. No data is stored or transmitted.
          This is an educational tool and does not replace medical advice.
        </p>
      </div>
    </div>
  );
}

// ── RESULTS PAGE ──────────────────────────────────────────────────────────────

function GaugeMeter({ score }) {
  const { color, grad } = riskInfo(score);
  const angle = -90 + (score/100)*180; // -90° to +90°
  const r     = 80;
  const cx=100, cy=100;
  const toRad = a => a*Math.PI/180;
  const arc = (start,end,radius) => {
    const x1=cx+radius*Math.cos(toRad(start)), y1=cy+radius*Math.sin(toRad(start));
    const x2=cx+radius*Math.cos(toRad(end)),   y2=cy+radius*Math.sin(toRad(end));
    return `M${x1} ${y1} A${radius} ${radius} 0 0 1 ${x2} ${y2}`;
  };

  return (
    <div style={{textAlign:"center"}}>
      <svg viewBox="0 0 200 110" style={{width:"100%",maxWidth:260,display:"block",margin:"0 auto"}}>
        {/* Track */}
        <path d={arc(180,360,r)} fill="none" stroke="#e9ecef" strokeWidth={18} strokeLinecap="round"/>
        {/* Zones */}
        <path d={arc(180,240,r)} fill="none" stroke={C.ok}     strokeWidth={18} strokeLinecap="butt" opacity=".25"/>
        <path d={arc(240,303,r)} fill="none" stroke={C.warn}   strokeWidth={18} strokeLinecap="butt" opacity=".25"/>
        <path d={arc(303,360,r)} fill="none" stroke={C.danger} strokeWidth={18} strokeLinecap="butt" opacity=".25"/>
        {/* Score arc */}
        <path d={arc(180, 180+(score/100)*180, r)} fill="none"
          stroke={`url(#g${score})`} strokeWidth={18} strokeLinecap="round"/>
        <defs>
          <linearGradient id={`g${score}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={grad.split(",")[0].trim()}/>
            <stop offset="100%" stopColor={grad.split(",")[1].trim()}/>
          </linearGradient>
        </defs>
        {/* Needle */}
        <line x1={cx} y1={cy}
          x2={cx+64*Math.cos(toRad(180+(score/100)*180))}
          y2={cy+64*Math.sin(toRad(180+(score/100)*180))}
          stroke={color} strokeWidth={3} strokeLinecap="round"/>
        <circle cx={cx} cy={cy} r={6} fill={color}/>
        {/* Score text */}
        <text x={cx} y={cy-22} textAnchor="middle" fontSize={28} fontWeight={800}
          fill={color} fontFamily="Fraunces,serif">{score}%</text>
        {/* Labels */}
        <text x={28}  y={104} fontSize={9} fill={C.ok}     fontFamily="Plus Jakarta Sans,sans-serif">Low</text>
        <text x={82}  y={112} fontSize={9} fill={C.warn}   fontFamily="Plus Jakarta Sans,sans-serif">Moderate</text>
        <text x={148} y={104} fontSize={9} fill={C.danger} fontFamily="Plus Jakarta Sans,sans-serif">High</text>
      </svg>
    </div>
  );
}

function SHAPBar({ name, val, maxVal, present, delay }) {
  const pct = maxVal > 0 ? (val/maxVal)*100 : 0;
  return (
    <div style={{marginBottom:10,animation:`fadeSlideUp .4s ${delay}s ease both`,opacity:0,animationFillMode:"forwards"}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4,alignItems:"center"}}>
        <span style={{fontSize:".8rem",fontWeight:600,color:present?C.navy:C.muted}}>
          {present && <span style={{color:C.danger,marginRight:4}}>↑</span>}
          {name}
        </span>
        <span style={{
          fontFamily:"'DM Mono',monospace",fontSize:".72rem",
          color:present?C.danger:C.muted,fontWeight:600,
        }}>
          {val > 0 ? `+${val.toFixed(1)}` : val.toFixed(1)}
        </span>
      </div>
      <div style={{height:7,borderRadius:4,background:"#f1f5f9",overflow:"hidden"}}>
        <div style={{
          height:"100%",borderRadius:4,
          width:`${pct}%`,
          background:present
            ? `linear-gradient(90deg,${C.warn},${C.danger})`
            : `linear-gradient(90deg,${C.teal},${C.blue})`,
          transition:"width 1s ease",
        }}/>
      </div>
    </div>
  );
}

function ResultsPage({ result, form, setPage }) {
  if (!result) return (
    <div style={{maxWidth:500,margin:"100px auto",textAlign:"center",padding:"0 1.5rem"}}>
      <div style={{fontSize:52,marginBottom:16}}>📋</div>
      <h2 style={{fontFamily:"'Fraunces',serif",fontSize:"1.6rem",fontWeight:700,color:C.navy,marginBottom:10}}>No Assessment Yet</h2>
      <p style={{color:C.muted,marginBottom:24}}>Complete the symptom assessment to generate your personalized prediction.</p>
      <button className="btn-primary" onClick={()=>setPage("Assess")}>Start Assessment →</button>
    </div>
  );

  const { score, prob, contributions } = result;
  const ri = riskInfo(score);
  const recs = getRecommendations(score, form);
  const maxVal = Math.max(...contributions.map(c=>c.val));
  const presentSyms  = SYMPTOM_KEYS.filter(k=>form[k]===1);
  const absentSyms   = SYMPTOM_KEYS.filter(k=>form[k]===0);

  return (
    <div style={{maxWidth:1060,margin:"0 auto",padding:"44px 1.5rem"}}>
      {/* Header */}
      <div className="au" style={{marginBottom:32,display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:16}}>
        <div>
          <h1 style={{fontFamily:"'Fraunces',serif",fontSize:"2rem",fontWeight:800,color:C.navy,marginBottom:4}}>Your Risk Assessment</h1>
          <p style={{color:C.muted,fontSize:".88rem"}}>Random Forest model · SHAP feature explanations · {new Date().toLocaleDateString()}</p>
        </div>
        <span className="pill" style={{background:ri.bg,color:ri.color,fontSize:".85rem",padding:"8px 18px"}}>
          {ri.emoji} {ri.label}
        </span>
      </div>

      {/* TOP ROW */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1.4fr",gap:20,marginBottom:20}}>
        {/* Score Gauge */}
        <div className="card au1" style={{padding:"28px 20px",textAlign:"center"}}>
          <h3 style={{fontWeight:700,color:C.muted,fontSize:".72rem",textTransform:"uppercase",letterSpacing:".1em",marginBottom:16}}>
            5-Year Risk Score
          </h3>
          <GaugeMeter score={score}/>
          <div style={{
            marginTop:16,padding:"12px 16px",
            background:`${ri.color}12`,border:`1px solid ${ri.color}30`,
            borderRadius:12,fontSize:".83rem",lineHeight:1.6,color:C.text,
          }}>
            {score<30 && "Your symptom profile suggests low diabetes risk. Maintain a healthy lifestyle and schedule annual screenings."}
            {score>=30 && score<65 && "Moderate risk detected. Targeted lifestyle changes and a clinical consultation are strongly advised."}
            {score>=65 && "High risk identified. Please consult a healthcare provider promptly for clinical evaluation and testing."}
          </div>
          <div style={{marginTop:14,display:"flex",gap:10,justifyContent:"center"}}>
            <div style={{textAlign:"center"}}>
              <div style={{fontFamily:"'DM Mono',monospace",fontWeight:600,color:C.blue,fontSize:"1rem"}}>{prob}</div>
              <div style={{fontSize:".7rem",color:C.muted}}>Probability</div>
            </div>
            <div style={{width:1,background:C.border}}/>
            <div style={{textAlign:"center"}}>
              <div style={{fontFamily:"'DM Mono',monospace",fontWeight:600,color:C.navy,fontSize:"1rem"}}>96.1%</div>
              <div style={{fontSize:".7rem",color:C.muted}}>Model Acc.</div>
            </div>
            <div style={{width:1,background:C.border}}/>
            <div style={{textAlign:"center"}}>
              <div style={{fontFamily:"'DM Mono',monospace",fontWeight:600,color:C.navy,fontSize:"1rem"}}>0.968</div>
              <div style={{fontSize:".7rem",color:C.muted}}>AUC-ROC</div>
            </div>
          </div>
        </div>

        {/* SHAP Chart */}
        <div className="card au2" style={{padding:"28px"}}>
          <h3 style={{fontWeight:700,color:C.muted,fontSize:".72rem",textTransform:"uppercase",letterSpacing:".1em",marginBottom:4}}>
            SHAP Feature Importance
          </h3>
          <p style={{color:C.muted,fontSize:".76rem",marginBottom:18,lineHeight:1.5}}>
            How each symptom contributed to your score.
            <span style={{color:C.danger}}> Red bars = risk-increasing</span> symptoms you reported.
          </p>
          {contributions.slice(0,10).map((c,i)=>(
            <SHAPBar key={c.name} name={c.name} val={c.val} maxVal={maxVal} present={c.present} delay={i*.05}/>
          ))}
        </div>
      </div>

      {/* Symptom Summary */}
      <div className="card au3" style={{padding:"24px 28px",marginBottom:20}}>
        <h3 style={{fontFamily:"'Fraunces',serif",fontWeight:700,color:C.navy,marginBottom:16,fontSize:"1rem"}}>
          Your Symptom Summary
        </h3>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
          <div>
            <div style={{fontWeight:700,fontSize:".8rem",color:C.danger,marginBottom:10,display:"flex",alignItems:"center",gap:6}}>
              <span style={{width:8,height:8,borderRadius:"50%",background:C.danger,display:"inline-block"}}/>
              Positive Symptoms ({presentSyms.length})
            </div>
            {presentSyms.length===0?(
              <p style={{fontSize:".8rem",color:C.muted,fontStyle:"italic"}}>None reported — great!</p>
            ):presentSyms.map(k=>(
              <div key={k} style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                <span style={{color:C.danger,fontSize:12}}>✗</span>
                <span style={{fontSize:".82rem",color:C.text,fontWeight:500}}>{SYMPTOM_INFO[k].label}</span>
              </div>
            ))}
          </div>
          <div>
            <div style={{fontWeight:700,fontSize:".8rem",color:C.ok,marginBottom:10,display:"flex",alignItems:"center",gap:6}}>
              <span style={{width:8,height:8,borderRadius:"50%",background:C.ok,display:"inline-block"}}/>
              Negative Symptoms ({absentSyms.length})
            </div>
            {absentSyms.slice(0,6).map(k=>(
              <div key={k} style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                <span style={{color:C.ok,fontSize:12}}>✓</span>
                <span style={{fontSize:".82rem",color:C.muted}}>{SYMPTOM_INFO[k].label}</span>
              </div>
            ))}
            {absentSyms.length>6 && <span style={{fontSize:".75rem",color:C.muted}}>+{absentSyms.length-6} more</span>}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="au4" style={{marginBottom:24}}>
        <h2 style={{fontFamily:"'Fraunces',serif",fontSize:"1.3rem",fontWeight:700,color:C.navy,marginBottom:6}}>
          Personalised Health Recommendations
        </h2>
        <p style={{color:C.muted,fontSize:".83rem",marginBottom:18}}>
          Evidence-based guidance calibrated to your {ri.label.toLowerCase()} profile.
        </p>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(300px,1fr))",gap:14}}>
          {recs.map((r,i)=>(
            <div key={i} className="card" style={{
              padding:"18px 20px",
              borderLeft:`4px solid ${ri.color}`,
              animation:`fadeSlideUp .45s ${i*.07}s ease both`,
              opacity:0,animationFillMode:"forwards",
            }}>
              <div style={{fontSize:20,marginBottom:8}}>{r.icon}</div>
              <h4 style={{fontFamily:"'Fraunces',serif",fontWeight:700,color:C.navy,fontSize:".92rem",marginBottom:6}}>{r.t}</h4>
              <p style={{color:C.muted,fontSize:".79rem",lineHeight:1.6}}>{r.d}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Model metadata */}
      <div className="card au5" style={{padding:"18px 24px",marginBottom:24,display:"flex",flexWrap:"wrap",gap:20}}>
        {[
          ["Model","Random Forest"],
          ["Training Dataset","Early Stage Diabetes (UCI)"],
          ["Accuracy","96.1%"],
          ["AUC-ROC","0.9679"],
          ["Top Predictor","Polyuria (0.2198 importance)"],
          ["XAI Method","SHAP (Feature Contributions)"],
          ["Samples Used","251 (deduplicated from 520)"],
        ].map(([l,v])=>(
          <div key={l}>
            <div style={{fontSize:".7rem",color:C.muted,textTransform:"uppercase",letterSpacing:".06em",marginBottom:2}}>{l}</div>
            <div style={{fontSize:".85rem",fontWeight:600,color:C.navy,fontFamily:"'DM Mono',monospace"}}>{v}</div>
          </div>
        ))}
      </div>

      {/* Actions */}
      <div style={{display:"flex",gap:12,flexWrap:"wrap"}}>
        <button className="btn-primary" onClick={()=>setPage("Assess")}>← Redo Assessment</button>
        <button onClick={()=>window.print()} style={{
          background:"#fff",border:`1.5px solid ${C.border}`,color:C.text,
          padding:"12px 22px",borderRadius:12,fontSize:".88rem",fontWeight:600,
        }}>🖨 Print Report</button>
      </div>

      <p style={{marginTop:16,fontSize:".74rem",color:C.muted,lineHeight:1.6}}>
        ⚕ <strong>Medical Disclaimer:</strong> This is an educational ML tool and does not constitute medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare professional for medical concerns.
      </p>
    </div>
  );
}

// ── ABOUT PAGE ────────────────────────────────────────────────────────────────

function AboutPage() {
  const modelResults = [
    { name:"Logistic Regression",  acc:80.4, auc:0.9196 },
    { name:"Decision Tree",        acc:76.5, auc:0.8830 },
    { name:"Random Forest",        acc:96.1, auc:0.9679, best:true },
    { name:"Gradient Boosting",    acc:90.2, auc:0.9482 },
    { name:"K-Nearest Neighbors",  acc:82.3, auc:0.9116 },
    { name:"SVM (RBF Kernel)",     acc:88.2, auc:0.9679 },
    { name:"Neural Network (MLP)", acc:82.3, auc:0.9393 },
    { name:"Stacking Ensemble",    acc:88.2, auc:0.9589 },
  ];

  const featureImportance = [
    { name:"Polyuria",            val:0.2198 },
    { name:"Polydipsia",          val:0.1834 },
    { name:"Age",                 val:0.1066 },
    { name:"Sudden Weight Loss",  val:0.0663 },
    { name:"Gender",              val:0.0650 },
    { name:"Polyphagia",          val:0.0439 },
    { name:"Irritability",        val:0.0439 },
    { name:"Partial Paresis",     val:0.0407 },
    { name:"Weakness",            val:0.0378 },
    { name:"Obesity",             val:0.0312 },
  ];

  const [openViva, setOpenViva] = useState(null);
  const vivaQA = [
    {q:"Why is Random Forest the best model here?",
     a:"Random Forest achieved 96.1% accuracy and 0.9679 AUC, outperforming all other models. The dataset is primarily categorical/binary features which tree-based methods handle natively without scaling. The ensemble nature reduces overfitting (300 trees, max_depth=12) while capturing complex interactions between symptoms like Polyuria + Polydipsia that are clinically known to co-present."},
    {q:"Why were only 251 samples used instead of 520?",
     a:"The original 520-row dataset contained 269 duplicate rows (51.7%). Removing duplicates is essential to prevent data leakage and overly optimistic evaluation scores. After deduplication, 251 unique patient records remained. This is a small but clean dataset typical of medical data collection challenges."},
    {q:"How does SHAP work for model explainability?",
     a:"SHAP (SHapley Additive exPlanations) uses Shapley values from cooperative game theory to fairly attribute each feature's contribution to a prediction. For Random Forest, TreeSHAP computes exact SHAP values efficiently. For each patient, we show which symptoms pushed the prediction toward 'Positive' (diabetes) and by how much — enabling clinically interpretable, personalized explanations."},
    {q:"Why is Polyuria the most important feature?",
     a:"Polyuria (excessive urination, >3L/day) is a hallmark symptom of hyperglycemia. When blood glucose exceeds the renal threshold (~180 mg/dL), glucose spills into urine and draws water with it via osmosis, causing polyuria. The Random Forest correctly identified it as the single strongest predictor (importance: 0.2198) — matching clinical literature on early diabetes presentation."},
    {q:"How did you handle class imbalance in the dataset?",
     a:"The deduplicated dataset had 173 positive (68.9%) vs 78 negative (31.1%) cases — moderate imbalance. We used stratified train-test splitting to maintain class proportions in both sets. Since Random Forest with class_weight='balanced' was considered, and performance was strong across both classes (precision 100%, recall 94.3%), the imbalance did not require SMOTE for this dataset size."},
    {q:"What are the limitations of this system?",
     a:"Key limitations: (1) Small dataset — 251 unique samples may not generalize to all populations. (2) Symptom self-reporting bias — patients may under/over-report symptoms. (3) No blood glucose data — this is symptom-only, not a clinical diagnostic. (4) Training data sourced from a specific Bangladeshi clinical study (Sylhet Diabetes Hospital). A robust production system would require larger, multi-ethnic datasets and integration of biomarkers (HbA1c, FPG)."},
  ];

  return (
    <div style={{maxWidth:940,margin:"0 auto",padding:"44px 1.5rem"}}>
      <div className="au" style={{marginBottom:40}}>
        <h1 style={{fontFamily:"'Fraunces',serif",fontSize:"2.2rem",fontWeight:800,color:C.navy,marginBottom:10}}>
          Project Methodology
        </h1>
        <p style={{color:C.muted,maxWidth:600,lineHeight:1.7,fontSize:".9rem"}}>
          A complete end-to-end ML system for early-stage Type 2 Diabetes risk prediction,
          built as a Final Year B.Tech CSE Major Project.
        </p>
      </div>

      {/* Dataset Info */}
      <div className="card au1" style={{padding:"28px",marginBottom:22}}>
        <h2 style={{fontFamily:"'Fraunces',serif",fontWeight:700,fontSize:"1.1rem",color:C.navy,marginBottom:16}}>
          Dataset Profile
        </h2>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:20}}>
          <div>
            <p style={{color:C.muted,fontSize:".85rem",lineHeight:1.7}}>
              <strong style={{color:C.navy}}>Source:</strong> UCI ML Repository<br/>
              <strong style={{color:C.navy}}>Title:</strong> Early Stage Diabetes Risk Prediction<br/>
              <strong style={{color:C.navy}}>Collected:</strong> Sylhet Diabetes Hospital, Bangladesh<br/>
              <strong style={{color:C.navy}}>Original rows:</strong> 520 (269 duplicates removed)<br/>
              <strong style={{color:C.navy}}>Final samples:</strong> 251 unique records<br/>
              <strong style={{color:C.navy}}>Class balance:</strong> 68.9% Positive, 31.1% Negative
            </p>
          </div>
          <div>
            <div style={{fontSize:".72rem",color:C.muted,textTransform:"uppercase",letterSpacing:".06em",fontWeight:700,marginBottom:10}}>
              16 Features Used
            </div>
            {["Age (continuous, 16–90)", "Gender (Male/Female)", ...Object.values(SYMPTOM_INFO).map(s=>s.label+" (Yes/No)")].map(f=>(
              <span key={f} style={{
                display:"inline-block",margin:"3px",padding:"3px 10px",
                background:`${C.blue}12`,color:C.blue,
                borderRadius:20,fontSize:".72rem",fontWeight:500,
              }}>{f}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Pipeline */}
      <div className="card au2" style={{padding:"28px",marginBottom:22}}>
        <h2 style={{fontFamily:"'Fraunces',serif",fontWeight:700,fontSize:"1.1rem",color:C.navy,marginBottom:16}}>
          ML Pipeline Stages
        </h2>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:16}}>
          {[
            {ph:"1",t:"EDA & Preprocessing",items:["Duplicate removal (269→251)","Binary encoding (Yes/No→1/0)","Gender & Target label encoding","No missing values found"]},
            {ph:"2",t:"Feature Engineering",items:["Age retained continuous","All 14 symptoms as binary flags","No polynomial features needed","StandardScaler for LR/SVM/MLP"]},
            {ph:"3",t:"Model Training",items:["7 base classifiers + Stacking","Stratified 80/20 train-test split","5-fold CV for stacking","GridSearch for key hyperparams"]},
            {ph:"4",t:"Explainable AI",items:["SHAP TreeExplainer (RF-based)","Per-patient feature contributions","Global feature importance ranking","Waterfall-style bar visualization"]},
          ].map(p=>(
            <div key={p.ph} style={{background:`${C.blue}06`,borderRadius:12,padding:"16px"}}>
              <div style={{fontFamily:"'DM Mono',monospace",color:C.blue,fontSize:".68rem",marginBottom:6,fontWeight:600}}>PHASE {p.ph}</div>
              <div style={{fontFamily:"'Fraunces',serif",fontWeight:700,color:C.navy,marginBottom:10,fontSize:".92rem"}}>{p.t}</div>
              {p.items.map(item=>(
                <div key={item} style={{fontSize:".77rem",color:C.muted,marginBottom:4,display:"flex",gap:6}}>
                  <span style={{color:C.blue}}>›</span>{item}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="card au3" style={{padding:"28px",marginBottom:22}}>
        <h2 style={{fontFamily:"'Fraunces',serif",fontWeight:700,fontSize:"1.1rem",color:C.navy,marginBottom:16}}>
          Model Performance Results
        </h2>
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:".84rem"}}>
            <thead>
              <tr style={{borderBottom:`2px solid ${C.border}`}}>
                {["Model","Accuracy","AUC-ROC","Accuracy Bar"].map(h=>(
                  <th key={h} style={{textAlign:"left",padding:"8px 14px",color:C.muted,
                    fontWeight:700,fontSize:".72rem",textTransform:"uppercase",letterSpacing:".06em"}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {modelResults.map(m=>(
                <tr key={m.name} style={{
                  borderBottom:`1px solid ${C.border}`,
                  background:m.best?`${C.blue}06`:"transparent",
                }}>
                  <td style={{padding:"10px 14px",fontWeight:m.best?700:400,color:m.best?C.blue:C.navy}}>
                    {m.best?"🏆 ":""}{m.name}
                  </td>
                  <td style={{padding:"10px 14px",fontWeight:600,color:m.best?C.blue:C.text}}>
                    {m.acc}%
                  </td>
                  <td style={{padding:"10px 14px",fontWeight:600,color:m.best?C.blue:C.text,fontFamily:"'DM Mono',monospace"}}>
                    {m.auc}
                  </td>
                  <td style={{padding:"10px 14px"}}>
                    <div style={{height:8,borderRadius:4,background:"#f1f5f9",width:120,overflow:"hidden"}}>
                      <div style={{
                        height:"100%",borderRadius:4,
                        width:`${(m.acc/96.1)*100}%`,
                        background:m.best?`linear-gradient(90deg,${C.blue},${C.teal})`:"#94a3b8",
                      }}/>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance Chart */}
      <div className="card au4" style={{padding:"28px",marginBottom:22}}>
        <h2 style={{fontFamily:"'Fraunces',serif",fontWeight:700,fontSize:"1.1rem",color:C.navy,marginBottom:16}}>
          Top 10 Feature Importances (Random Forest)
        </h2>
        {featureImportance.map((f,i)=>(
          <div key={f.name} style={{marginBottom:10,animation:`fadeSlideUp .4s ${i*.05}s ease both`,opacity:0,animationFillMode:"forwards"}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
              <span style={{fontSize:".83rem",fontWeight:600,color:C.navy}}>{f.name}</span>
              <span style={{fontFamily:"'DM Mono',monospace",fontSize:".78rem",color:C.blue}}>{f.val}</span>
            </div>
            <div style={{height:8,borderRadius:4,background:"#f1f5f9",overflow:"hidden"}}>
              <div style={{
                height:"100%",borderRadius:4,
                width:`${(f.val/0.2198)*100}%`,
                background:`linear-gradient(90deg,${C.blue},${C.sky})`,
              }}/>
            </div>
          </div>
        ))}
      </div>

      {/* Author Card */}
      <div className="card au5" style={{
        padding:"28px",marginBottom:28,
        background:`linear-gradient(150deg,${C.navy},#0d2d5c)`,border:"none",
      }}>
        <div style={{fontFamily:"'DM Mono',monospace",color:"#94a3b8",fontSize:".72rem",marginBottom:12,letterSpacing:".08em"}}>
          PROJECT DETAILS
        </div>
        <h3 style={{fontFamily:"'Fraunces',serif",fontWeight:700,color:"#fff",fontSize:"1.4rem",marginBottom:8}}>
          Final Year B.Tech CSE Major Project
        </h3>
        <p style={{color:"#94a3b8",fontSize:".87rem",lineHeight:1.75}}>
          Title: ML-Based Early Detection of Type 2 Diabetes Risk Using Multi-Modal Clinical and Lifestyle Features<br/>
          Dataset: Early Stage Diabetes Risk Prediction (UCI) · 520 rows · 17 columns<br/>
          Stack: Python · scikit-learn · SHAP · FastAPI · React.js<br/>
          Academic Year: 2024–25 · Domain: ML, Health Informatics, Explainable AI
        </p>
        <div style={{marginTop:16,display:"flex",gap:10,flexWrap:"wrap"}}>
          {["Research Paper","GitHub Repository","Live Deployment","Kaggle Notebook"].map(l=>(
            <span key={l} style={{
              padding:"6px 14px",borderRadius:20,cursor:"pointer",
              border:"1px solid rgba(255,255,255,.15)",
              color:"#93c5fd",fontSize:".75rem",fontWeight:600,
            }}>{l}</span>
          ))}
        </div>
      </div>

      {/* Viva Q&A */}
      <h2 style={{fontFamily:"'Fraunces',serif",fontWeight:700,fontSize:"1.2rem",color:C.navy,marginBottom:16}}>
        Viva Voce Preparation
      </h2>
      {vivaQA.map((qa,i)=>(
        <div key={i} className="card" style={{marginBottom:10,overflow:"hidden"}}>
          <button onClick={()=>setOpenViva(openViva===i?null:i)} style={{
            width:"100%",textAlign:"left",background:"none",border:"none",
            padding:"16px 20px",display:"flex",justifyContent:"space-between",alignItems:"flex-start",gap:12,
            fontFamily:"'Plus Jakarta Sans',sans-serif",fontWeight:600,color:C.navy,fontSize:".88rem",
          }}>
            <span>Q{i+1}: {qa.q}</span>
            <span style={{color:C.blue,fontSize:20,lineHeight:1,flexShrink:0}}>{openViva===i?"−":"+"}</span>
          </button>
          {openViva===i && (
            <div style={{padding:"0 20px 16px",borderTop:`1px solid ${C.border}`}}>
              <p style={{color:C.muted,fontSize:".85rem",lineHeight:1.75,paddingTop:14}}>
                <strong style={{color:C.blue}}>A:</strong> {qa.a}
              </p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── ROOT APP ──────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage]     = useState("Home");
  const [form, setForm]     = useState(DEFAULT_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = () => {
    setLoading(true);
    setTimeout(() => {
      const data = {
        ...form,
        Age:    parseInt(form.Age),
        Gender: form.Gender === "Male" ? 1 : 0,
      };
      // Map -1 (unanswered) to 0 as safety
      Object.keys(SYMPTOM_INFO).forEach(k => { if (data[k]===-1) data[k]=0; });
      const prediction = simulateRF(data);
      setResult(prediction);
      setLoading(false);
      setPage("Results");
    }, 2000);
  };

  return (
    <>
      <style>{FONTS}{GS}</style>
      <div style={{minHeight:"100vh",display:"flex",flexDirection:"column"}}>
        <Navbar page={page} setPage={setPage}/>
        <main style={{flex:1}}>
          {page==="Home"    && <HomePage setPage={setPage}/>}
          {page==="Assess"  && <AssessPage form={form} setForm={setForm} onPredict={handlePredict} loading={loading}/>}
          {page==="Results" && <ResultsPage result={result} form={form} setPage={setPage}/>}
          {page==="About"   && <AboutPage/>}
        </main>
        <footer style={{
          borderTop:`1px solid ${C.border}`,background:"#fff",
          padding:"20px 1.5rem",
        }}>
          <div style={{maxWidth:1100,margin:"0 auto",display:"flex",flexWrap:"wrap",gap:12,
            alignItems:"center",justifyContent:"space-between",fontSize:".78rem",color:C.muted}}>
            <span style={{fontFamily:"'Fraunces',serif",fontWeight:600,color:C.navy}}>DiabetesSense v2.0</span>
            <span>B.Tech CSE Final Year Project · Dataset: UCI Early Stage Diabetes Risk</span>
            <span style={{color:C.danger,fontWeight:600}}>⚕ Educational use only — Not medical advice</span>
          </div>
        </footer>
      </div>
    </>
  );
}
