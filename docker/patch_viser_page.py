"""Browser tab of the GUI: viser's page hardcodes <title>Viser</title> and its own icon and never changes them.

Sets the title ModelForge and our icon (docker/favicon.svg), and adds a script that follows a hidden element of the
GUI (#mf-status, `App._status_html`): data-title becomes the tab title, data-state (idle / running / failed) picks
the icon (running: a green dot, failed: a red one), and a change of data-notice ("<n>|<text>", `App._notice`: a queue
entry or the queue ended) shows a browser notification (the browser asks for permission after the first click in the
page; notifications need https or localhost).

Clicks on an img[data-mf-click="<uuid>"] (gui/tiepoints.py: the Tie point window) are sent to the hidden text input
of that uuid as "fx,fy,time" (fractions of the image's width / height).

UPLOADER: streaming, resumable uploads (modelforge/gui/upload.py has the protocol): a file picked in an element with
data-mf-upload="<target>" (gui/upload.widget) is sent in 1 MiB parts with at most 32 MiB beyond the server's
acknowledgement, through viser's own message sender (window.__mfV, set by docker/patch_viser_camera.py, which also
routes the acknowledgements here: window.__mfAck); a stalled transfer (a dropped connection) starts again and the
server answers how much it has, so it resumes; picking the same file after a reload resumes too.

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


UPLOADER = """<script id="mf-upload">(()=>{
const CH=1048576,WIN=32*CH,ups={};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const id=()=>crypto.randomUUID?crypto.randomUUID():Date.now()+"-"+Math.random().toString(16).slice(2);
const send=m=>{const v=window.__mfV;if(!v||!v.sendMessage)return false;try{v.sendMessage(m);return true}catch(e){return false}};
const mb=b=>(b/1e6).toFixed(b<1e8?1:0);
window.__mfAck=n=>{const u=ups[n.transfer_uuid];if(!u)return false;u.onAck(n.transferred_bytes);return true};
async function one(file,target,show){
 const parts=Math.ceil(file.size/CH),src="mf-upload|"+target+"|"+file.lastModified;
 let u={acked:-1,sent:0,last:Date.now(),wake:null,err:false},t0=Date.now(),t1=0;
 const start=()=>{if(u.tid)delete ups[u.tid];u.tid=id();ups[u.tid]=u;u.acked=-1;u.started=Date.now();
  send({type:"FileTransferStartUpload",source_component_uuid:src,transfer_uuid:u.tid,filename:file.name,
   mime_type:file.type||"",part_count:parts,size_bytes:file.size})};
 u.onAck=b=>{u.last=Date.now();if(b<0){u.err=true}else{if(u.acked<0){u.sent=b;if(b>0&&!t1)t1=b}
  u.acked=Math.max(u.acked,b)}if(u.wake){const w=u.wake;u.wake=null;w()}};
 const waitAck=ms=>new Promise(r=>{u.wake=r;setTimeout(()=>{if(u.wake===r){u.wake=null;r()}},ms)});
 const status=()=>{const s=(Date.now()-t0)/1000,rate=(Math.max(u.acked,0)-t1)/Math.max(s,.1);
  show(file.name+": "+mb(Math.max(u.acked,0))+" / "+mb(file.size)+" MB ("+(100*Math.max(u.acked,0)/Math.max(file.size,1)).toFixed(0)+" %)"
   +(rate>0?", "+mb(rate)+" MB/s":"")+(t1?" · resumed at "+mb(t1)+" MB":""))};
 start();
 while(!u.err&&u.acked<file.size){
  if(u.acked<0){await waitAck(1000);if(u.acked<0&&Date.now()-u.started>5000){show(file.name+": waiting for the server …");start()}continue}
  if(u.sent<file.size&&u.sent-u.acked<WIN){const i=Math.floor(u.sent/CH);
   const buf=await file.slice(i*CH,Math.min(file.size,(i+1)*CH)).arrayBuffer();
   if(!send({type:"FileTransferPart",source_component_uuid:"mf-upload",transfer_uuid:u.tid,part_index:i,content:new Uint8Array(buf)})){await sleep(1000);continue}
   u.sent=Math.min(file.size,(i+1)*CH);if(i%8===0)status();continue}
  await waitAck(1000);status();
  if(Date.now()-u.last>8000){show(file.name+": no answer, resuming …");start()}}
 delete ups[u.tid];
 show(u.err?file.name+": refused or failed (see the message in the tab)":file.name+": uploaded ("+mb(file.size)+" MB in "+((Date.now()-t0)/1000).toFixed(0)+" s)");
 return !u.err}
let busy=Promise.resolve();
document.addEventListener("change",e=>{const t=e.target;if(!t||!t.matches||!t.matches("input[data-mf-upload]"))return;
 const box=t.closest("[data-mf-upload-box]"),out=box&&box.querySelector(".mf-upload-status"),target=t.dataset.mfUpload;
 const show=s=>{if(out)out.textContent=s};
 for(const f of Array.from(t.files))busy=busy.then(()=>one(f,target,show)).catch(x=>show(String(x)));t.value=""},true);
document.addEventListener("click",e=>{const t=e.target;if(!t||!t.matches||!t.matches("img[data-mf-click]"))return;
 const r=t.getBoundingClientRect(),v=window.__mfV;if(!v||!v.sendMessage||!r.width)return;
 v.sendMessage({type:"GuiUpdateMessage",uuid:t.dataset.mfClick,updates:{value:((e.clientX-r.left)/r.width).toFixed(6)+","+
 ((e.clientY-r.top)/r.height).toFixed(6)+","+Date.now()}})},true);
window.addEventListener("beforeunload",e=>{if(Object.keys(ups).length){e.preventDefault();e.returnValue=""}});
})()</script>"""


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
    page = page.replace("</head>", (SCRIPT % js).replace("\n", "") + UPLOADER.replace("\n", "") + "</head>", 1)
    assert "mf-status" in page
    open(index, "w").write(page)


if __name__ == "__main__":
    main(*sys.argv[1:])
