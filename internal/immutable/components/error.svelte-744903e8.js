import{S as h,i as j,s as w,k as b,q as E,a as N,e as F,l as d,m as P,r as R,h as m,c as S,b as _,F as $,u as q,X as H,O as y}from"../chunks/index-af87de52.js";import{p as z}from"../chunks/stores-cab3d7fc.js";function O(p){let r,a=p[0].error.frame+"",f;return{c(){r=b("pre"),f=E(a)},l(l){r=d(l,"PRE",{});var s=P(r);f=R(s,a),s.forEach(m)},m(l,s){_(l,r,s),$(r,f)},p(l,s){s&1&&a!==(a=l[0].error.frame+"")&&q(f,a)},d(l){l&&m(r)}}}function X(p){let r,a=p[0].error.stack+"",f;return{c(){r=b("pre"),f=E(a)},l(l){r=d(l,"PRE",{});var s=P(r);f=R(s,a),s.forEach(m)},m(l,s){_(l,r,s),$(r,f)},p(l,s){s&1&&a!==(a=l[0].error.stack+"")&&q(f,a)},d(l){l&&m(r)}}}function A(p){let r,a=p[0].status+"",f,l,s,c=p[0].error.message+"",k,v,u,n,t=p[0].error.frame&&O(p),i=p[0].error.stack&&X(p);return{c(){r=b("h1"),f=E(a),l=N(),s=b("pre"),k=E(c),v=N(),t&&t.c(),u=N(),i&&i.c(),n=F()},l(e){r=d(e,"H1",{});var o=P(r);f=R(o,a),o.forEach(m),l=S(e),s=d(e,"PRE",{});var C=P(s);k=R(C,c),C.forEach(m),v=S(e),t&&t.l(e),u=S(e),i&&i.l(e),n=F()},m(e,o){_(e,r,o),$(r,f),_(e,l,o),_(e,s,o),$(s,k),_(e,v,o),t&&t.m(e,o),_(e,u,o),i&&i.m(e,o),_(e,n,o)},p(e,[o]){o&1&&a!==(a=e[0].status+"")&&q(f,a),o&1&&c!==(c=e[0].error.message+"")&&q(k,c),e[0].error.frame?t?t.p(e,o):(t=O(e),t.c(),t.m(u.parentNode,u)):t&&(t.d(1),t=null),e[0].error.stack?i?i.p(e,o):(i=X(e),i.c(),i.m(n.parentNode,n)):i&&(i.d(1),i=null)},i:H,o:H,d(e){e&&m(r),e&&m(l),e&&m(s),e&&m(v),t&&t.d(e),e&&m(u),i&&i.d(e),e&&m(n)}}}function B(p,r,a){let f;return y(p,z,l=>a(0,f=l)),[f]}class I extends h{constructor(r){super(),j(this,r,B,A,w,{})}}export{I as default};
