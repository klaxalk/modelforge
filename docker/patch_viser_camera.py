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

View-dependent splat colours (SH below; View → Splat colours, modelforge/gui/data.py `pack_splats`): viser draws one
RGB per Gaussian (band 0). A splat buffer may carry spherical-harmonic coefficients after its 8 words per Gaussian: the
first Gaussian's reserved word (3, which the renderer overwrites with the group index) holds 0x4D46 << 16 | degree << 8 |
K, then K words per Gaussian follow the N x 8 block (a float16 scale, then int8 coefficients, coefficient-major with
interleaved channels). window.__mfMerge (the merge of all splat objects) keeps the 8-word layout for the sorter and the
main texture and gathers the coefficients into a second texture (mfShTex, ceil(K / 4) texels per Gaussian, zeros for
objects without them); the vertex shader adds the bands for the direction from the camera to the Gaussian, as gsplat
does. Buffers without the tag are drawn as before.

window.__mfV (viser's message sender for the page's uploader) is set every frame, and upload acknowledgements of
the streaming uploads go to window.__mfAck. Splat buffer updates are staged until the sort worker has ordered them (SPLAT below): no gap, no stale order.

Re-applying is safe: the loader's edit and the script are replaced, so this runs on an image patched by the older
patch_viser_keys.sh too. Fails if viser's loader or the hook changed.

Usage: python patch_viser_camera.py VISER_INDEX_HTML
"""
import re
import sys

HOOK = "qf((e,t)=>{let n=a.cameraControl,r=j.current;if(n===null||r.size===0||!n.enabled)return;let i=2*t,o=50*Yt.DEG2RAD*t;"
# the hook's own argument renamed, so `e` is the viewer context again (NL(e): viser's scene root transform)
NEW_HOOK = ("qf((mfX,t)=>{window.__mfV=a;let n=a.cameraControl,r=j.current;let mfE=document.getElementById(`mf-move-speed`),"
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
    # streaming uploads (docker/patch_viser_page.py UPLOADER): their acknowledgements go to the page's uploader
    ("case`FileTransferPartAck`:f({", "case`FileTransferPartAck`:if(window.__mfAck&&window.__mfAck(n))return;f({"),
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
     "e.fill(0),e.set(P.gaussianBuffer),s.current.textureBuffer.needsUpdate=!0,mfShTex(s.current.material,P.gaussianBuffer,P.numGaussians,a)}t.mutable.current.requestRender()}};"
     "m({updateBuffer:p.gaussianBuffer,updateGroupIndices:p.groupIndices,updateNumGroups:p.numGroups,gen:G})}"
     "else if(g){"),
)
# View-dependent colours (see the module doc). GLSL on single lines: no // comments. No arrays or loops: every
# coefficient byte is read with constant indices (dynamically indexed local arrays, the first version, worked in
# software WebGL but drew nothing on the user's GPU), and only the texels a degree needs are fetched.
def _sh_glsl():
    def coef(k):  # coefficient k (0..14) as r, g, b: byte 2 + 3k + ch of the Gaussian's words
        parts = []
        for ch in range(3):
            b = 2 + 3 * k + ch
            w = b // 4
            parts.append(f"mfB(w{w // 4}.{'xyzw'[w % 4]}, {8 * (b % 4)}u)")
        return f"vec3({', '.join(parts)})"

    fetch = "texelFetch(mfShTex, ivec2((b + {o}u) % uint(ts.x), (b + {o}u) / uint(ts.x)), 0)"
    return (
        "float mfB(uint w, uint s) { float v = float((w >> s) & 255u); return v > 127.0 ? v - 256.0 : v; } "
        "vec3 mfSh(uint idx, vec3 d) { ivec2 ts = textureSize(mfShTex, 0); uint b = idx * mfShT; "
        f"uvec4 w0 = {fetch.format(o=0)}; float x = d.x, y = d.y, z = d.z; "
        f"vec3 r = 0.4886025119029199 * (-y * {coef(0)} + z * {coef(1)} - x * {coef(2)}); "
        f"if (mfDeg > 1) {{ uvec4 w1 = {fetch.format(o=1)}; float xx = x * x, yy = y * y, zz = z * z; "
        f"r += 1.0925484305920792 * x * y * {coef(3)} - 1.0925484305920792 * y * z * {coef(4)} "
        f"+ 0.31539156525252005 * (2.0 * zz - xx - yy) * {coef(5)} - 1.0925484305920792 * x * z * {coef(6)} "
        f"+ 0.5462742152960396 * (xx - yy) * {coef(7)}; "
        f"if (mfDeg > 2) {{ uvec4 w2 = {fetch.format(o=2)}; "
        f"r += -0.5900435899266435 * y * (3.0 * xx - yy) * {coef(8)} + 2.890611442640554 * x * y * z * {coef(9)} "
        f"- 0.4570457994644658 * y * (4.0 * zz - xx - yy) * {coef(10)} "
        f"+ 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * {coef(11)} "
        f"- 0.4570457994644658 * x * (4.0 * zz - xx - yy) * {coef(12)} + 1.445305721320277 * z * (xx - yy) * {coef(13)} "
        f"- 0.5900435899266435 * x * (xx - 3.0 * yy) * {coef(14)}; }} }} "
        "return r * (unpackHalf2x16(w0.x).x / 127.0); } ")


SH_FN = _sh_glsl()
# the SH texture of a material (created, resized or refilled; a 1 x 1 dummy without coefficients: an integer sampler
# must be bound to an integer texture); module scope, so three.js' DataTexture (Gi) and formats are in reach
SH_TEX = ("function mfShTex(f,e,r,n){let S=e.mfSh,T=S?e.mfShT:1,u=f.uniforms,o=u.mfShTex&&u.mfShTex.value,"
          "w=S?Math.max(1,Math.min(r*T,n)):1,h=S?Math.max(1,Math.ceil(r*T/w)):1;"
          "if(o&&o.image.width===w&&o.image.height===h){let D=o.image.data;D.fill(0),S&&D.set(S),o.needsUpdate=!0}"
          "else{window.__mfShInfo=`${w}x${h} texels, ${T} per Gaussian, degree ${S?e.mfDeg:0}, ${r} Gaussians`;"
          "S&&window.__mfDiag&&window.__mfDiag(`sh texture ${window.__mfShInfo}`);"
          "let D=new Uint32Array(w*h*4);S&&D.set(S);let X=new Gi(D,w,h,le,L);X.internalFormat=`RGBA32UI`,"
          "X.needsUpdate=!0,o&&o.dispose();if(u.mfShTex)u.mfShTex.value=X;else{u.mfShTex={value:X};"
          "f.addEventListener(`dispose`,()=>u.mfShTex.value.dispose())}}"
          "u.mfShT?u.mfShT.value=T:u.mfShT={value:T};let G=S?e.mfDeg:0;u.mfDeg?u.mfDeg.value=G:u.mfDeg={value:G};"
          "f.uniformsNeedUpdate=!0}")
MERGE = ("function bne(e){let t=0;for(let n of Object.values(e))t+=n.length;let n=t/8,r=new Uint32Array(t),"
         "i=new Uint32Array(n),a=0;for(let[t,n]of Object.values(e).entries()){i.fill(t,a/8,(a+n.length)/8),r.set(n,a);"
         "for(let e=0;e<n.length;e+=8)r[a+e+3]=t;a+=n.length}return{numGaussians:n,gaussianBuffer:r,"
         "numGroups:Object.keys(e).length,groupIndices:i}}")
SH = (
    (MERGE, "function bne(e){return window.__mfMerge(e)}"),
    ("function pz(e,t,n){", SH_TEX + "function pz(e,t,n){"),
    ("let f=new fz;return f.fog=!0,", "let f=new fz;mfShTex(f,e,r,n);return f.fog=!0,"),
    ("e.fill(0),e.set(p.gaussianBuffer),s.current.textureBuffer.needsUpdate=!0,",
     "e.fill(0),e.set(p.gaussianBuffer),s.current.textureBuffer.needsUpdate=!0,"
     "mfShTex(s.current.material,p.gaussianBuffer,p.numGaussians,a),"),
    ("uniform float transitionInState;",
     "uniform float transitionInState; uniform usampler2D mfShTex; uniform uint mfShT; uniform int mfDeg;"),
    ("mat4 getGroupTransform(uint i) {", SH_FN + "mat4 getGroupTransform(uint i) {"),
    ("// Throw the Gaussian off the screen if it's too close, too far, or too small.",
     "if (mfDeg > 0) vRgba.rgb = clamp(vRgba.rgb + mfSh(sortedIndex, normalize(transpose(mat3(T_camera_group)) "
     "* c_cam.xyz)), 0.0, 1.0); // Throw the Gaussian off the screen if it's too close, too far, or too small."),
)
EDITS = ('.replaceAll("`KeyQ`","`KeyF`").replaceAll("`KeyE`","`KeyR`")'
         f'.replace("{HOOK}","{NEW_HOOK}")'
         + "".join(f'.{"replaceAll" if old == SPLAT_SEND else "replace"}("{old}","{new}")' for old, new in SPLAT + SH))

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


MERGE_JS = """<script id="mf-sh">window.__mfMerge=e=>{
let B=Object.values(e),n=0,T=0,D=0,P=[];
for(const b of B){const h=b.length>=8?b[3]>>>0:0;let K=0,d=0;if(h>>>16===0x4D46){K=h&255;d=h>>>8&255}
 const g=b.length/(8+K);P.push([b,g,K]);n+=g;if(K){T=Math.max(T,Math.ceil(K/4));D=Math.max(D,d)}}
const r=new Uint32Array(n*8),i=new Uint32Array(n),S=T?new Uint32Array(n*T*4):null;let a=0;
P.forEach(([b,g,K],t)=>{i.fill(t,a,a+g);r.set(b.subarray(0,g*8),a*8);for(let k=0;k<g;k++)r[(a+k)*8+3]=t;
 if(K){const s=b.subarray(g*8);if(K===4*T)S.set(s,a*4*T);else for(let k=0;k<g;k++)S.set(s.subarray(k*K,k*K+K),(a+k)*4*T)}
 a+=g});
if(S){r.mfSh=S;r.mfShT=T;r.mfDeg=D}
return{numGaussians:n,gaussianBuffer:r,numGroups:B.length,groupIndices:i}};</script>"""
# WebGL diagnostics to the GUI log (gui/app.py `_page_diag`, through the hidden input data-diag names): the GPU and its
# limits, the SH texture's shape, GL errors (polled each second), lost contexts, three.js errors and warnings
DIAG_JS = """<script id="mf-diag">(()=>{const q=[],seen=new Set();let sent=0,gl=null,lost=0;
window.__mfDiag=s=>{s=String(s).slice(0,400);if(!seen.has(s)&&seen.size<200){seen.add(s);q.push(s)}};
for(const k of ["error","warn"]){const f=console[k].bind(console);console[k]=(...a)=>{f(...a);try{const s=a.map(String).join(" ");
 if(/THREE|WebGL|shader|GL_|context|program/i.test(s)&&!/computeBoundingSphere|reversed depth|Clock/.test(s))window.__mfDiag(k+": "+s)}catch(e){}}}
setInterval(()=>{const c=document.querySelector("canvas");
 if(!gl&&c){gl=c.getContext("webgl2");if(gl){const x=gl.getExtension("WEBGL_debug_renderer_info");
  window.__mfDiag("gpu "+(x?gl.getParameter(x.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER))+" | max texture "+
   gl.getParameter(gl.MAX_TEXTURE_SIZE)+" | vertex texture units "+gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS)+
   " | "+navigator.userAgent);
  c.addEventListener("webglcontextlost",()=>{lost++;window.__mfDiag("WebGL context lost ("+lost+")")});
  c.addEventListener("webglcontextrestored",()=>window.__mfDiag("WebGL context restored"))}}
 if(gl){const e=gl.getError();if(e)window.__mfDiag("GL error 0x"+e.toString(16)+(window.__mfShInfo?" (sh "+window.__mfShInfo+")":""))}
 const el=document.getElementById("mf-move-speed"),u=el&&el.dataset.diag,a=window.__mfV;
 while(q.length&&u&&a&&a.sendMessage&&sent<300){a.sendMessage({type:"GuiUpdateMessage",uuid:u,updates:{value:(++sent)+" "+q.shift()}})}},1000)})();</script>"""


def main(index):
    page = open(index).read()
    page, n = re.subn(re.escape(DECODE) + r"[^\n]*?;(?=\s*const run=)", lambda _: DECODE + EDITS + ";", page, count=1)
    assert n == 1, "viser's bundle loader not found"
    page = re.sub(r'<script id="mf-fly">.*?</script>', "", page, flags=re.S)
    page = re.sub(r'<script id="mf-(sh|diag)">.*?</script>', "", page, flags=re.S)
    page = page.replace("</head>", FLY.replace("\n", "") + MERGE_JS.replace("\n", "") + DIAG_JS.replace("\n", "")
                        + "</head>", 1)
    assert NEW_HOOK in page and "__mfFly" in page and "__mfMerge" in page
    open(index, "w").write(page)


if __name__ == "__main__":
    main(*sys.argv[1:])
