"""viser's camera in the browser, changed in its client bundle (zstd-compressed inside index.html, so its loader edits
the code after unpacking; each pattern occurs once, in the camera controls' per-frame hook):
 - Q / E move down / up; the user wants F / R (the movement key set and the elevate calls);
 - the move speed: viser's fixed 2 m/s (`let i=2*t` with t the frame time) becomes the GUI's View → Move speed,
   read from a hidden element (#mf-move-speed, data-speed) while a key is held; 5 m/s until the page has one;
 - View → Orbit: data-orbit (deg/s, 0 = off) turns the view every frame about its look-at point, as the arrow keys
   do, so the mouse wheel and drags keep working while it orbits (the server-driven orbit overwrote them);
 - View → Fly: data-fly (the clock: path id, epoch, t0, speed, loop, look-at distance, hold) and #mf-fly-path
   (data-path: path times, camera centres, wxyz quaternions of R_wc in the GUI's world) let the page fly the path
   itself, every frame, through window.__mfFly (a script added to the page): the same cubic Hermite / slerp as
   gui.data.FlightPath, put into three.js' frame with viser's scene root transform. A drag, wheel or held movement
   key on the 3D view hands the camera to the user until `hold` seconds after the last one and pauses the flight:
   the page stops its clock at once and reports "pause:k:t" / then "resume:k:t" as a GuiUpdateMessage to the
   hidden input data-fly names (`report`), so the server pauses the clock (`paused`, for every browser, the
   slider and the photo window) and continues it from t.

Splat buffer updates are staged until the sort worker has ordered them (SPLAT below): no gap, no stale order.

Re-applying is safe: the loader's edit and the script are replaced, so this runs on an image patched by the older
patch_viser_keys.sh too. Fails if viser's loader or the hook changed.

Usage: python patch_viser_camera.py VISER_INDEX_HTML
"""
import re
import sys

HOOK = "qf((e,t)=>{let n=a.cameraControl,r=j.current;if(n===null||r.size===0||!n.enabled)return;let i=2*t,o=50*Yt.DEG2RAD*t;"
# the hook's own argument renamed, so `e` is the viewer context again (NL(e): viser's scene root transform)
NEW_HOOK = ("qf((mfX,t)=>{let n=a.cameraControl,r=j.current;let mfE=document.getElementById(`mf-move-speed`),"
            "mfO=+(mfE?.dataset.orbit||0);if(n!==null&&n.enabled&&window.__mfFly?.(n,()=>NL(e),r.size>0,a))return;"
            "if(n===null||r.size===0&&!mfO||!n.enabled)return;mfO&&n.rotate(mfO*Yt.DEG2RAD*Math.min(t,.1),0,!0);"
            "let i=(window.__mfSpeed=+(mfE?.dataset.speed||window.__mfSpeed||5))*t,o=50*Yt.DEG2RAD*t;")
DECODE = "const code=td.decode(heap.subarray(up+ss,up+sz))"
# Splat updates without a gap (the user saw changed LoD cells flash to the background, then flicker): viser applied a
# changed splat buffer at once, with a draw order the sort worker had not computed for it yet (a new buffer size:
# empty, so nothing drawn; same size: the old order on the new layout). Now the worker tags each sort with the buffer
# generation it sorted (`gen`), and an update (same group count) only stages the new buffer: the old one is drawn,
# in its own order, until the worker's first sort of the new one arrives; then buffer and order are applied together.
SPLAT_SEND = "m({updateBuffer:p.gaussianBuffer,updateGroupIndices:p.groupIndices,updateNumGroups:p.numGroups})"
SPLAT = (
    # the worker: the generation of the buffer it sorts, in every result
    ("self.postMessage({sortedIndices:o},[o.buffer])", "self.postMessage({sortedIndices:o,gen:self.__mfg|0},[o.buffer])"),
    ("e.setBuffer(n.updateBuffer,n.updateGroupIndices),r=!0",
     "e.setBuffer(n.updateBuffer,n.updateGroupIndices),self.__mfg=n.gen|0,r=!0"),
    # viser's own update paths (another group count): shown at once, as before, as a new generation
    (SPLAT_SEND, "m({updateBuffer:p.gaussianBuffer,updateGroupIndices:p.groupIndices,updateNumGroups:p.numGroups,"
                 "gen:s.mfShown=s.mfGen=(s.mfGen|0)+1}),s.mfPending=null"),
    # a sort result: the staged buffer goes in with its first order; results for other generations are dropped
    ("o.current.onmessage=e=>{let n=e.data.sortedIndices;s.current&&n.length===s.current.numGaussians&&(",
     "o.current.onmessage=e=>{let n=e.data.sortedIndices,G=e.data.gen|0,P=s.mfPending;if(P&&G===P.gen){"
     "s.mfPending=null;s.mfShown=G;P.apply()}if(G!==(s.mfShown|0))return;"
     "s.current&&n.length===s.current.numGaussians&&("),
    # an update with the same group count: staged
    ("else if(h&&s.current){if(g){",
     "else if(h&&s.current){if(p.numGroups===s.current.numGroups){let G=(s.mfGen|0)+1,P=p;s.mfGen=G;"
     "s.mfPending={gen:G,apply:()=>{if(P.numGaussians!==s.current.numGaussians){let e=s.current;"
     "s.current=pz(P.gaussianBuffer,P.numGroups,a);setTimeout(()=>{e.textureBuffer.dispose(),e.geometry.dispose(),"
     "e.material.dispose(),e.textureT_camera_groups.dispose()},500);u.current=new Float32Array;"
     "s.current.material.uniforms.transitionInState.value=1;s.current.material.uniforms.numGaussians.value="
     "P.numGaussians;s.current.textureBuffer.needsUpdate=!0;f()}else{let e=s.current.textureBuffer.image.data;"
     "e.fill(0),e.set(P.gaussianBuffer),s.current.textureBuffer.needsUpdate=!0}t.mutable.current.requestRender()}};"
     "m({updateBuffer:p.gaussianBuffer,updateGroupIndices:p.groupIndices,updateNumGroups:p.numGroups,gen:G})}"
     "else if(g){"),
)
EDITS = ('.replaceAll("`KeyQ`","`KeyF`").replaceAll("`KeyE`","`KeyR`")'
         f'.replace("{HOOK}","{NEW_HOOK}")'
         + "".join(f'.{"replaceAll" if old == SPLAT_SEND else "replace"}("{old}","{new}")' for old, new in SPLAT))

FLY = """<script id="mf-fly">(()=>{
let P=null,Pk=null,C=null,Ck=null,t0=0,touch=-1e9,hold=false,frozen=null,resumed=0,seq=0;
const now=()=>performance.now(),mark=()=>{touch=now()};
const on3d=e=>e.target&&e.target.tagName==="CANVAS";
addEventListener("pointerdown",e=>{on3d(e)&&mark()},true);
addEventListener("pointermove",e=>{e.buttons&&on3d(e)&&mark()},true);
addEventListener("wheel",e=>{on3d(e)&&mark()},{capture:true,passive:true});
const path=()=>{const el=document.getElementById("mf-fly-path"),s=el?el.dataset.path||"":"";
 if(s!==Pk){Pk=s;P=null;if(s){const d=JSON.parse(s),n=d.t.length,t=d.t,p=d.p,m=new Float64Array(3*n);
  for(let k=0;k<n;k++){const a=k>0?k-1:0,b=k<n-1?k+1:n-1;for(let j=0;j<3;j++)m[3*k+j]=(p[3*b+j]-p[3*a+j])/(t[b]-t[a]);}
  P={id:d.id,t:t,p:p,q:d.q,m:m};}}return P};
const pose=(P,t)=>{const T=P.t,n=T.length;let lo=0,hi=n-1;while(hi-lo>1){const c=(lo+hi)>>1;if(T[c]<=t)lo=c;else hi=c;}
 const k=lo,h=T[k+1]-T[k],u=Math.min(1,Math.max(0,(t-T[k])/h)),u2=u*u,u3=u2*u;
 const a=2*u3-3*u2+1,b=u3-2*u2+u,c=-2*u3+3*u2,d=u3-u2,p=P.p,m=P.m,x=[0,0,0];
 for(let j=0;j<3;j++)x[j]=a*p[3*k+j]+b*h*m[3*k+j]+c*p[3*k+3+j]+d*h*m[3*k+3+j];
 const q0=P.q.slice(4*k,4*k+4);let q1=P.q.slice(4*k+4,4*k+8),dt=q0[0]*q1[0]+q0[1]*q1[1]+q0[2]*q1[2]+q0[3]*q1[3];
 if(dt<0){q1=q1.map(v=>-v);dt=-dt}let q;
 if(dt>0.9995)q=q0.map((v,i)=>v+u*(q1[i]-v));else{const th=Math.acos(dt),s=Math.sin(th);
  q=q0.map((v,i)=>(Math.sin((1-u)*th)*v+Math.sin(u*th)*q1[i])/s)}
 const l=Math.hypot(...q),w=q[0]/l,X=q[1]/l,Y=q[2]/l,Z=q[3]/l;
 return [x,[2*(X*Z+w*Y),2*(Y*Z-w*X),1-2*(X*X+Y*Y)],[-2*(X*Y-w*Z),-(1-2*(X*X+Z*Z)),-2*(Y*Z+w*X)]]};
const report=(a,k,t)=>{C.report&&a.sendMessage&&a.sendMessage({type:"GuiUpdateMessage",uuid:C.report,
 updates:{value:k+":"+(++seq)+":"+t}})};
window.__mfFly=(n,root,keys,a)=>{const el=document.getElementById("mf-move-speed"),s=el?el.dataset.fly||"":"";
 if(!s){C=null;Ck=null;hold=false;frozen=null;return false}
 if(s!==Ck){Ck=s;C=JSON.parse(s);t0=now();if(!C.paused&&!hold)frozen=null}
 const P=path();if(!P||P.id!==C.path)return false;if(keys)mark();
 const D=P.t[P.t.length-1],clock=()=>{const t=C.paused?C.t0:C.t0+C.speed*(now()-t0)/1000;
  return C.loop?((t%D)+D)%D:Math.min(t,D)};
 if(now()-touch<1000*C.hold){if(!hold){hold=true;frozen=clock();report(a,"pause",frozen)}
  a.requestRender&&a.requestRender();return false}
 if(hold){hold=false;resumed=now();report(a,"resume",frozen)}
 if(frozen!==null&&now()-resumed>3000)frozen=null;
 const t=frozen!==null?frozen:clock();
 const [x,f,up]=pose(P,t),M=root().elements;
 const pt=v=>[M[0]*v[0]+M[4]*v[1]+M[8]*v[2]+M[12],M[1]*v[0]+M[5]*v[1]+M[9]*v[2]+M[13],M[2]*v[0]+M[6]*v[1]+M[10]*v[2]+M[14]];
 const dir=v=>[M[0]*v[0]+M[4]*v[1]+M[8]*v[2],M[1]*v[0]+M[5]*v[1]+M[9]*v[2],M[2]*v[0]+M[6]*v[1]+M[10]*v[2]];
 const p=pt(x),g=pt([x[0]+f[0]*C.dist,x[1]+f[1]*C.dist,x[2]+f[2]*C.dist]),u=dir(up);
 n.camera.up.set(u[0],u[1],u[2]).normalize();n.updateCameraUp();n.setLookAt(p[0],p[1],p[2],g[0],g[1],g[2],false);
 a.requestRender&&a.requestRender();return true};
})()</script>"""


def main(index):
    page = open(index).read()
    page, n = re.subn(re.escape(DECODE) + r"[^\n]*?;(?=\s*const run=)", lambda _: DECODE + EDITS + ";", page, count=1)
    assert n == 1, "viser's bundle loader not found"
    page = re.sub(r'<script id="mf-fly">.*?</script>', "", page, flags=re.S)
    page = page.replace("</head>", FLY.replace("\n", "") + "</head>", 1)
    assert NEW_HOOK in page and "__mfFly" in page
    open(index, "w").write(page)


if __name__ == "__main__":
    main(*sys.argv[1:])
