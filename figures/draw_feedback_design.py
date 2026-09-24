"""Draw Figure 1 as editable SVG and vector PDF from the same primitives.

Requires Python 3, ReportLab, and DejaVu Sans. The generated SVG contains named
layers and live text, with no raster assets or converted text outlines.
"""
from pathlib import Path
from contextlib import contextmanager
from html import escape
import argparse
import math

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import toColor as HexColor

W, H = 1440, 1080
INK = '#203247'
MUTED = '#596B7C'
LINE = '#D7E0E8'
FREE = '#718096'
ACTUAL = '#087F83'
DONOR = '#C27A35'
PALE = '#F3F6F9'
TEAL_PALE = '#E9F5F2'
ORANGE_PALE = '#FFF3E7'


class Drawing:
    def __init__(self, out):
        self.out = out
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" xml:space="preserve" width="7.5in" height="{7.5 * H / W}in" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
                    '<title id="title">Controlled diagnosis of hidden feedback errors</title>',
                    '<desc id="desc">Matched training objectives; one pose-preserving feedback replacement with actual or donor guidance; three free continuations under identical actions; paired comparisons of future position error. Pose readout is a measurement branch, not a transition input.</desc>']
        self.pdf = canvas.Canvas(str(out / 'feedback_design.pdf'), pagesize=(540, 540 * H / W), invariant=1,
                                 initialFontName='FigureSans', initialFontSize=22)
        self.pdf.setTitle('Figure 1: Controlled diagnosis of hidden feedback errors')
        self.pdf.setAuthor('')
        self.pdf.setCreator('Vector figure source')
        self.pdf.scale(540 / W, 540 / W)
        self.bounds = []
        self.text_sizes = []

    @contextmanager
    def layer(self, name):
        self.svg.append(f'<g id="{name}" inkscape:groupmode="layer" inkscape:label="{name.replace("-", " ")}">')
        yield
        self.svg.append('</g>')

    def rect(self, x, y, w, h, fill='white', stroke=LINE, r=8, lw=1.8):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
        c = self.pdf
        c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor(stroke)); c.setLineWidth(lw)
        c.roundRect(x, H-y-h, w, h, r, stroke=1, fill=1)

    def path(self, pts, color=INK, lw=2.2, dash=False, arrow=False):
        path = ' '.join(('M' if i == 0 else 'L') + f'{x},{y}' for i, (x, y) in enumerate(pts))
        da = ' stroke-dasharray="6 5"' if dash else ''
        self.svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{lw}" stroke-linejoin="round"{da}/>')
        c = self.pdf; c.setStrokeColor(HexColor(color)); c.setLineWidth(lw)
        c.setDash([6, 5] if dash else [])
        p = c.beginPath(); p.moveTo(pts[0][0], H-pts[0][1])
        for x, y in pts[1:]: p.lineTo(x, H-y)
        c.drawPath(p); c.setDash([])
        if arrow:
            (x0, y0), (x, y) = pts[-2:]
            a = math.atan2(y-y0, x-x0); size = 8
            q = [(x,y), (x-size*math.cos(a)+4*math.sin(a), y-size*math.sin(a)-4*math.cos(a)),
                 (x-size*math.cos(a)-4*math.sin(a), y-size*math.sin(a)+4*math.cos(a))]
            self.svg.append(f'<polygon points="{" ".join(f"{u},{v}" for u,v in q)}" fill="{color}"/>')
            p = c.beginPath(); p.moveTo(q[0][0], H-q[0][1])
            for u,v in q[1:]: p.lineTo(u,H-v)
            p.close(); c.setFillColor(HexColor(color)); c.drawPath(p, stroke=0, fill=1)

    def circle(self, x, y, r, fill='white', stroke=INK, lw=2):
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
        c = self.pdf; c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor(stroke)); c.setLineWidth(lw)
        c.circle(x, H-y, r, stroke=1, fill=1)

    def text(self, x, y, value, size=28, color=INK, bold=False, anchor='start', italic=False):
        font = 'FigureBold' if bold else 'FigureItalic' if italic else 'FigureSans'
        width = pdfmetrics.stringWidth(value, font, size)
        left = x if anchor == 'start' else x-width if anchor == 'end' else x-width/2
        assert 0 <= left and left+width <= W, (value, left, width)
        self.bounds.append((left, y-size, left+width, y+4, value))
        self.text_sizes.append(size)
        self.svg.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" font-style="{"oblique" if italic else "normal"}" text-anchor="{anchor}" fill="{color}">{escape(value)}</text>')
        c = self.pdf; c.setFont(font, size); c.setFillColor(HexColor(color)); c.drawString(left, H-y, value)

    def math(self, x, y, parts, size=28, color=INK, anchor='start'):
        """Native editable math: tuples are (base, subscript, superscript).

        Variables are oblique; word-valued branch/selector labels are upright.
        This avoids Unicode lookalike sub/superscript letters and preserves text.
        """
        def width(value, fs, it=False):
            return pdfmetrics.stringWidth(value, 'FigureItalic' if it else 'FigureSans', fs)
        def script_runs(value):
            if value in {'free', 'act', 'don', 'pos', 'A', 'B', 'C', 'D', 'eval', 'min'}:
                return [(value, False)]
            return [(char, char.isalpha()) for char in value]
        def script_width(value):
            return sum(width(v, size*.66, it) for v,it in script_runs(value))
        def script_text(sx, sy, value):
            for v,it in script_runs(value):
                self.text(sx,sy,v,size=size*.66,color=color,italic=it)
                sx+=width(v,size*.66,it)
        layouts=[]
        for part in parts:
            if isinstance(part,str):
                layouts.append((part,width(part,size),None));continue
            base,sub,sup=part
            it=len(base)==1 and base.isalpha()
            bw=width(base,size,it)
            sw=max(script_width(sub),script_width(sup)) if (sub or sup) else 0
            layouts.append((part,bw+sw+1,(bw,it)))
        total=sum(v[1] for v in layouts)
        if anchor=='middle':x-=total/2
        elif anchor=='end':x-=total
        for value,w,extra in layouts:
            if extra is None:self.text(x,y,value,size=size,color=color)
            else:
                base,sub,sup=value;bw,it=extra
                self.text(x,y,base,size=size,color=color,italic=it)
                if sub:script_text(x+bw+1,y+size*.22,sub)
                if sup:script_text(x+bw+1,y-size*.46,sup)
            x+=w

    def token(self, x, y, value, color=INK, r=29):
        self.circle(x, y, r, stroke=color)
        self.text(x, y+8, value, size=25, color=color, anchor='middle')

    def module(self, x, y, label, w=52, h=48, color=INK, fill=PALE):
        self.rect(x-w/2, y-h/2, w, h, fill=fill, stroke=color, r=5)
        self.math(x, y+8, [(label,'','')], size=28, color=color, anchor='middle')

    def finish(self):
        self.svg.append('</svg>')
        (self.out / 'feedback_design.svg').write_text('\n'.join(self.svg)+'\n')
        self.pdf.showPage(); self.pdf.save()
        import json
        (self.out / 'feedback_design_qa.json').write_text(json.dumps({'canvas': [W,H], 'text_bounds': self.bounds, 'min_font_paper_pt_including_math_scripts': min(self.text_sizes)*396/W, 'supporting_label_paper_pt': 22*396/W, 'raster_assets': 0}, indent=2)+'\n')


def draw(out):
    d = Drawing(out)
    d.rect(0, 0, W, H, fill='white', stroke='white', r=0, lw=0)

    def stage(x, y, letter, title):
        d.circle(x+16,y-10,16,fill=INK,stroke=INK,lw=0)
        d.text(x+16,y-1,letter,size=23,color='white',bold=True,anchor='middle')
        d.text(x+43,y,title,size=29,bold=True)

    def token(x, y, branch, color):
        d.circle(x,y,35,stroke=color,lw=2.4)
        d.math(x,y+9,[('z','5',branch)],size=30,color=color,anchor='middle')

    with d.layer('title-and-question'):
        d.text(20,36,'Hold the task readout fixed. Test the feedback.',size=36,bold=True)
        d.text(20,73,'One intervention at action 5; compare prediction of the same executed future at action 25.',size=27,color=MUTED)
        d.path([(20,94),(1420,94)],color=LINE,lw=1.5)

    with d.layer('A-shared-prediction'):
        stage(20,139,'A','Predict once')
        d.text(20,178,'Observed history',size=26,color=MUTED)
        d.rect(20,198,240,100,fill=PALE,stroke=LINE,r=8)
        d.math(140,234,[('H','0',''),' = (',('E','',''),'(',('o','−10',''),'),'],size=27,anchor='middle')
        d.math(140,275,[('E','',''),'(',('o','−5',''),'), ',('E','',''),'(',('o','0',''),'))'],size=27,anchor='middle')
        d.path([(140,298),(140,330)],arrow=True)
        d.rect(93,330,94,64,fill=PALE,stroke=INK,r=7)
        d.math(140,370,[('F','θ','')],size=34,anchor='middle')
        d.math(237,347,[('U','0','')],size=29,anchor='middle')
        d.path([(235,354),(235,362),(187,362)],arrow=True)
        d.path([(140,394),(140,449)],arrow=True)
        d.circle(140,484,35,stroke=INK,lw=2.4)
        d.math(140,494,[('z','5','')],size=34,anchor='middle')
        d.text(140,552,'First prediction',size=24,anchor='middle',color=MUTED)
        d.text(20,625,'Matched models',size=26,bold=True)
        d.text(20,662,'Transformer / GRU',size=24,color=MUTED)
        d.text(20,697,'3 pools × 3 objectives',size=24,color=MUTED)
        d.text(20,732,'18 models in total',size=24,color=MUTED)
        # The same predicted token starts every branch; this spine carries z_5.
        d.path([(175,484),(288,484)],lw=2.2)
        d.path([(288,268),(288,568)],lw=2.2)
        d.circle(288,484,3.5,fill=INK,stroke=INK,lw=0)

    with d.layer('B-single-feedback-intervention'):
        stage(314,139,'B','Change feedback')
        d.text(314,178,'One correction; six outputs fixed',size=25,color=MUTED)
        # Free path bypasses projection and norm matching.
        d.path([(288,268),(679,268)],color=FREE,arrow=True)
        d.text(474,242,'Free: unchanged',size=26,color=FREE,anchor='middle')
        token(714,268,'free',FREE)

        # One common displacement norm. The spine z_5 also enters reconstruction.
        d.rect(526,375,111,232,fill=PALE,stroke=LINE,r=7)
        d.circle(581,268,3,fill=FREE,stroke=FREE,lw=0)
        d.path([(581,268),(581,375)],color=FREE,arrow=True)
        d.math(609,348,[('z','5','')],size=26,color=FREE,anchor='middle')
        d.text(581,407,'Match',size=25,bold=True,anchor='middle')
        d.text(581,439,'norm',size=25,bold=True,anchor='middle')
        d.math(581,493,[('m','','')],size=33,anchor='middle')
        d.text(581,541,'Unscale',size=24,anchor='middle',color=MUTED)
        d.math(581,578,['+ ',('z','5','')],size=26,anchor='middle')
        for y,color,pale,branch,label in [(418,ACTUAL,TEAL_PALE,'act','Actual observation'),(568,DONOR,ORANGE_PALE,'don','Donor observation')]:
            d.text(409,y-99,label,size=24,color=color,anchor='middle')
            d.rect(338,y-82,142,40,fill=pale,stroke=pale,r=6)
            d.math(409,y-54,[('E','',''),'(',('o','5',branch),')'],size=28,color=color,anchor='middle')
            d.path([(409,y-42),(409,y-27)],color=color,dash=True,arrow=True)
            d.path([(288,y),(338,y)],color=color,arrow=True)
            d.rect(338,y-27,142,54,fill=pale,stroke=color,r=6)
            d.text(409,y+9,'Project',size=27,color=color,anchor='middle')
            d.path([(480,y),(526,y)],color=color,arrow=True)
            d.math(501,y-12,[('d','',branch)],size=23,color=color,anchor='middle')
            d.path([(637,y),(679,y)],color=color,arrow=True)
            token(714,y,branch,color)
        # Constraint and dose are explicitly stated in standardized coordinates.
        d.rect(314,635,440,108,fill=PALE,stroke=PALE,r=8)
        d.math(534,670,[('J','g',''),' ',('d','',''),' = 0;  ',('x','',''),' + ',('d','',''),' ∈ ',('ℛ','g',''),';  ',('g','',''),' = ',('g','A','')],size=24,anchor='middle')
        d.text(534,703,'Original ReLU activation region',size=24,color=MUTED,anchor='middle')
        d.math(534,733,[('m','',''),': minimum norm over 6 directions'],size=24,color=MUTED,anchor='middle')

    with d.layer('C-independent-rollouts'):
        stage(794,139,'C','Roll each branch forward')
        d.text(794,178,'Same actions; no further observation inputs',size=25,color=MUTED)
        for y,color,pale,branch in [(268,FREE,PALE,'free'),(418,ACTUAL,TEAL_PALE,'act'),(568,DONOR,ORANGE_PALE,'don')]:
            # Three slots encode H_5^b. The only changed slot is the latest token.
            d.path([(749,y),(795,y)],color=color,arrow=True)
            d.math(855,y-40,[('H','5',branch)],size=28,color=color,anchor='middle')
            for j,(time,fill,stroke) in enumerate([('−5',PALE,LINE),('0',PALE,LINE),('5',pale,color)]):
                d.rect(795+43*j,y-24,39,48,fill=fill,stroke=stroke,r=4)
                d.text(814+43*j,y+8,time,size=24,color=color if j==2 else MUTED,anchor='middle')
            d.path([(920,y),(958,y)],color=color,arrow=True)
            d.rect(958,y-30,135,60,fill=pale,stroke=color,r=7)
            d.math(1025,y-1,[('F','θ','')],size=29,color=color,anchor='middle')
            d.text(1025,y+23,'4 steps',size=24,color=color,anchor='middle')
            d.math(1025,y-63,[('U','t','')],size=27,anchor='middle',color=MUTED)
            d.path([(1025,y-54),(1025,y-30)],color=MUTED,arrow=True)
            d.path([(1093,y),(1133,y)],color=color,arrow=True)
            d.math(1171,y+9,[('z','25',branch)],size=29,color=color,anchor='middle')
            d.path([(1211,y),(1250,y)],color=color,arrow=True)
            d.rect(1250,y-25,67,50,fill='white',stroke=color,r=5)
            d.math(1283,y+8,[('g','pos','')],size=27,color=color,anchor='middle')
            d.path([(1317,y),(1354,y)],color=color,arrow=True)
            d.math(1390,y+9,[('L','25',branch)],size=29,color=color,anchor='middle')
        d.rect(794,635,626,108,fill=PALE,stroke=PALE,r=8)
        d.math(1107,670,[('H','5','b'),' = (',('E','',''),'(',('o','−5',''),'), ',('E','',''),'(',('o','0',''),'), ',('z','5','b'),')'],size=27,anchor='middle')
        d.text(1107,703,'Shift + append; three aligned action blocks',size=24,color=MUTED,anchor='middle')
        d.math(1107,733,[('t','',''),' ∈ {5, 10, 15, 20};  ',('b','',''),' ∈ {free, act, don}'],size=24,color=MUTED,anchor='middle')

    with d.layer('D-preserved-measurement-and-outcome'):
        d.path([(20,768),(1420,768)],color=LINE,lw=1.5)
        stage(20,811,'D','Same current readout')
        d.math(362,861,[('g','A',''),'(',('z','5','free'),') = ',('g','A',''),'(',('z','5','act'),') = ',('g','A',''),'(',('z','5','don'),')'],size=31,anchor='middle')
        d.text(362,900,'Equality at insertion, within each model.',size=24,color=MUTED,anchor='middle')
        d.text(795,811,'Terminal block-position error',size=29,bold=True)
        d.math(795,857,[('L','25','b'),' = ‖',('g','pos',''),'(',('z','25','b'),') − ',('p','25',''),'‖₂²'],size=30)
        d.text(795,895,'Actual − free: useful correction?',size=25,color=ACTUAL)
        d.text(795,927,'Actual − donor: source sensitivity?',size=25,color=ACTUAL)
        d.text(20,931,'Core: 256 recipients · 6 groups · 4 reference streams',size=24,color=MUTED)

    with d.layer('E-extensions'):
        d.path([(20,953),(1420,953)],color=LINE,lw=1.5)
        stage(20,993,'E','Test the boundary')
        d.text(20,1033,'Measurement · diagnosis · decisions',size=22,color=MUTED)
        d.path([(452,975),(452,1054)],color=LINE,lw=1.5)
        d.text(475,988,'Reserved evaluator',size=22,bold=True)
        d.math(475,1023,[('g','A',''),', ',('g','C',''),'; score ',('g','D','')],size=25)
        d.text(475,1052,'8 directions',size=22,color=MUTED)
        d.text(475,1075,'256 fresh recipients',size=22,color=MUTED)
        d.path([(723,975),(723,1054)],color=LINE,lw=1.5)
        d.text(746,988,'Incremental diagnosis',size=24,bold=True)
        d.text(746,1023,'T0 probe + errors + norm²',size=22)
        d.text(746,1052,'Free error: T0 − T1',size=22,color=MUTED)
        d.text(746,1075,'256 calibration; 512 test',size=22,color=MUTED)
        d.path([(1107,975),(1107,1054)],color=LINE,lw=1.5)
        d.text(1130,988,'Matched decision',size=23,bold=True)
        d.text(1130,1023,'T0 / T1; 32 candidates',size=22)
        d.text(1130,1052,'8 directions',size=22,color=MUTED)
        d.text(1130,1075,'512 fresh recipients',size=22,color=MUTED)
    d.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--font-dir', type=Path, default=Path('/usr/share/fonts/truetype/dejavu'))
    args = p.parse_args()
    for name, filename in [('FigureSans', 'DejaVuSans.ttf'), ('FigureBold', 'DejaVuSans-Bold.ttf'), ('FigureItalic', 'DejaVuSans-Oblique.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(args.font_dir / filename)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(args.output_dir)


if __name__ == '__main__':
    main()
