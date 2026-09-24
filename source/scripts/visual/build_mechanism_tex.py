"""Compile the single Markdown manuscript into ICLR LaTeX; preserve result sources."""
from pathlib import Path
import re, subprocess, json, hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper/iclr2027'
p=ROOT/'paper/MECHANISM_MANUSCRIPT.md';s=p.read_text().split('## Source map')[0]
title=s.splitlines()[0][2:]
abstract=s.split('## Abstract\n\n',1)[1].split('## 1.',1)[0].strip()
body='## 1.'+s.split('## 1.',1)[1]
body=re.sub(r'^(#{2,3}) \d+(?:\.\d+)?\.? ',r'\1 ',body,flags=re.M)
figures={
'feedback_design':('feedback-design',"Design of the independent-goal prediction confirmation in PushT. (A) Three continuation objectives are compared in six matched groups. $E$ includes the observation projector; $F_\\theta$ includes the action encoder, temporal predictor, and prediction projector. (B) Actual-trajectory ($\\mathrm{act}$) and randomly assigned donor-trajectory ($\\mathrm{don}$) observations guide fixed-region projections. Their standardized corrections $d_c^s$ are rescaled with a shared $m$, computed over three objectives $c$ and two sources $s$ for each group, recipient, and reference stream. Model indices are suppressed in the diagram. The complete six-dimensional readout $g$ is preserved within each model; numerical checks are given in Appendix~\\ref{app:projection}. (C) Each branch starts with its own $H_5^b$ and uses the same subsequent action windows. Time $t$ counts primitive actions; $u_t=(a_t,\\ldots,a_{t+4})$ is a five-action block. No further observation is supplied. (D) $g_{\\mathrm{pos}}$ selects the block's position from $g$. Its squared Euclidean error against the same true endpoint is averaged before the five paired comparisons. Secondary full resets are omitted. Results appear in Figure~\\ref{fig:confirmation}. Nearest-donor and action-selection studies use separate protocols (Appendices~\\ref{app:nearest} and~\\ref{app:decisions})."),
'reversal':('reversal','Better coordinate supervision reverses its advantage under self-generated feedback. Physical-label training improves endpoint position prediction relative to the coordinate teacher with observed history, but worsens it in free rollout. Both estimates use the same 128 independent goals, six model groups, and four reference streams; bars show prespecified secondary 95\\% intervals. Negative differences favor physical labels.'),
'confirmation':('confirmation','Independent confirmation of the feedback mechanism. Every row compares actual-trajectory guidance with free rollout or external-donor guidance at action 25. All five primary 99\\% intervals lie below zero. Small points show all six model groups, including the physical-label donor comparison favoring the donor. All guidance branches preserve pose and use matched displacement.')}
for filename,text in [('abstract.tex',abstract),('body.tex',body)]:
 text=text.replace('−',r'\ensuremath{-}').replace('×',r'$\times$').replace('–','--')
 def figure(m):
  name=m[1];label,caption=figures[name]
  return '\\begin{figure}[tb]\n\\centering\\includegraphics[width=\\linewidth]{figures/'+name+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:'+label+'}\n\\end{figure}'
 text=re.sub(r'!\[[^]]+\]\(iclr2027/figures/([^.)]+)\.png\)',figure,text)
 result=subprocess.run(['pandoc','--from=markdown+raw_tex+tex_math_dollars','--to=latex','--shift-heading-level-by=-1'],input=text,text=True,capture_output=True,check=True).stdout
 # Keep manuscript cross-references joined by a nonbreaking space, not a printed tilde.
 result=result.replace(r'\textasciitilde{}\ref{',r'~\ref{')
 (OUT/filename).write_text(result)
main=OUT/'main.tex';text=main.read_text();text=re.sub(r'\\title\{[^\n]+\}',lambda _:r'\title{'+title.replace(': ', ':'+r'\\'+' ',1).replace(' in Latent World Models', r'\\'+' in Latent World Models',1)+'}',text);text=text.replace('\\input{confirmation.tex}\n','').replace('\\input{discussion.tex}\n','');text=re.sub(r'\\hypersetup\{hidelinks(?:,pdftitle=\{[^\n]+\})?\}',lambda _:r'\hypersetup{hidelinks,pdftitle={'+title+'}}',text);main.write_text(text)
(OUT/'build_source.json').write_text(json.dumps({'manuscript_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'title':title,'body_source':'All main-text sections derive from MECHANISM_MANUSCRIPT.md; appendix.tex is maintained explicitly.'},indent=2)+'\n')
