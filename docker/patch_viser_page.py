"""Browser tab of the GUI: viser's page hardcodes <title>Viser</title> and its own icon and never changes them.

Sets the title ModelForge and our icon (docker/favicon.svg), and adds a script that follows a hidden element of the
GUI (#mf-status, `App._status_html`): data-title becomes the tab title, data-state (idle / running / failed) picks
the icon (running: a green dot, failed: a red one), and a change of data-notice ("<n>|<text>", `App._notice`: a queue
entry or the queue ended) shows a browser notification (the browser asks for permission after the first click in the
page; notifications need https or localhost).

Usage: python patch_viser_page.py VISER_INDEX_HTML FAVICON_SVG
"""
import re
import sys
from urllib.parse import quote

BADGES = {"idle": "", "running": "#2ecc71", "failed": "#ff4d4f"}

SCRIPT = """<script>(()=>{const I=%s;let s0="",n0=null;const f=()=>{const e=document.getElementById("mf-status");if(!e)return;
const t=e.dataset.title||"ModelForge";if(document.title!==t)document.title=t;const s=e.dataset.state||"idle";
if(s!==s0){s0=s;const l=document.getElementById("mf-icon");if(l)l.href=I[s]||I.idle;}
const n=e.dataset.notice||"";if(n0===null){n0=n;}else if(n!==n0){n0=n;const k=n.indexOf("|");
if(k>=0&&window.Notification&&Notification.permission==="granted"){try{new Notification("ModelForge",
{body:n.slice(k+1),icon:I.idle});}catch(x){}}}};
document.addEventListener("click",()=>{if(window.Notification&&Notification.permission==="default")
Notification.requestPermission();},{once:true});
new MutationObserver(f).observe(document.documentElement,{subtree:true,childList:true,attributes:true,
attributeFilter:["data-title","data-state","data-notice"]});setInterval(f,2000);})()</script>"""


def icon(svg, colour):
    if colour:
        svg = svg.replace("</svg>", f'<circle cx="50" cy="50" r="11" fill="{colour}" stroke="#17212d" '
                                    'stroke-width="4"/></svg>')
    return "data:image/svg+xml," + quote(re.sub(r"\s+", " ", svg).strip(), safe="=:/'")


def main(index, favicon):
    page = open(index).read()
    svg = open(favicon).read()
    icons = {k: icon(svg, c) for k, c in BADGES.items()}
    page, n = re.subn(r'<link\s+rel="icon"[^>]*/>', f'<link rel="icon" id="mf-icon" href="{icons["idle"]}" />',
                      page, count=1)
    assert n == 1, "viser's icon link not found"
    page, n = re.subn(r"<title>[^<]*</title>", "<title>ModelForge</title>", page, count=1)
    assert n == 1, "viser's title not found"
    js = "{" + ",".join(f'{k}:"{v}"' for k, v in icons.items()) + "}"
    page = page.replace("</head>", (SCRIPT % js).replace("\n", "") + "</head>", 1)
    assert "mf-status" in page
    open(index, "w").write(page)


if __name__ == "__main__":
    main(*sys.argv[1:])
