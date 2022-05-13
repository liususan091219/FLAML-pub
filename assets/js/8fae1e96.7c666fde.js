"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[5882],{3905:function(e,t,n){n.d(t,{Zo:function(){return u},kt:function(){return d}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var l=r.createContext({}),s=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},u=function(e){var t=s(e.components);return r.createElement(l.Provider,{value:t},e.children)},f={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},p=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,u=c(e,["components","mdxType","originalType","parentName"]),p=s(n),d=a,g=p["".concat(l,".").concat(d)]||p[d]||f[d]||o;return n?r.createElement(g,i(i({ref:t},u),{},{components:n})):r.createElement(g,i({ref:t},u))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=p;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:a,i[1]=c;for(var s=2;s<o;s++)i[s]=n[s];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}p.displayName="MDXCreateElement"},9668:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return c},contentTitle:function(){return l},metadata:function(){return s},toc:function(){return u},default:function(){return p}});var r=n(7462),a=n(3366),o=(n(7294),n(3905)),i=["components"],c={sidebar_label:"switch_head_auto",title:"nlp.huggingface.switch_head_auto"},l=void 0,s={unversionedId:"reference/nlp/huggingface/switch_head_auto",id:"reference/nlp/huggingface/switch_head_auto",isDocsHomePage:!1,title:"nlp.huggingface.switch_head_auto",description:"AutoSeqClassificationHead Objects",source:"@site/docs/reference/nlp/huggingface/switch_head_auto.md",sourceDirName:"reference/nlp/huggingface",slug:"/reference/nlp/huggingface/switch_head_auto",permalink:"/FLAML/docs/reference/nlp/huggingface/switch_head_auto",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/reference/nlp/huggingface/switch_head_auto.md",tags:[],version:"current",frontMatter:{sidebar_label:"switch_head_auto",title:"nlp.huggingface.switch_head_auto"},sidebar:"referenceSideBar",previous:{title:"suggest",permalink:"/FLAML/docs/reference/default/suggest"},next:{title:"trainer",permalink:"/FLAML/docs/reference/nlp/huggingface/trainer"}},u=[{value:"AutoSeqClassificationHead Objects",id:"autoseqclassificationhead-objects",children:[{value:"from_model_type_and_config",id:"from_model_type_and_config",children:[],level:4}],level:2}],f={toc:u};function p(e){var t=e.components,n=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,r.Z)({},f,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h2",{id:"autoseqclassificationhead-objects"},"AutoSeqClassificationHead Objects"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"class AutoSeqClassificationHead()\n")),(0,o.kt)("p",null,"This is a class for getting classification head class based on the name of the LM\ninstantiated as one of the ClassificationHead classes of the library when\ncreated with the ",(0,o.kt)("inlineCode",{parentName:"p"},"AutoSeqClassificationHead.from_model_type_and_config")," method."),(0,o.kt)("p",null,"This class cannot be instantiated directly using ",(0,o.kt)("inlineCode",{parentName:"p"},"__init__()")," (throws an error)."),(0,o.kt)("h4",{id:"from_model_type_and_config"},"from","_","model","_","type","_","and","_","config"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"@classmethod\ndef from_model_type_and_config(cls, model_type: str, config: transformers.PretrainedConfig)\n")),(0,o.kt)("p",null,"Instantiate one of the classification head classes from the mode_type and model configuration."),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},"model_type"),' - A string, which desribes the model type, e.g., "electra".'),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},"config")," - The huggingface class of the model's configuration.")),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Example"),":"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'from transformers import AutoConfig\nmodel_config = AutoConfig.from_pretrained("google/electra-base-discriminator")\nAutoSeqClassificationHead.from_model_type_and_config("electra", model_config)\n')))}p.isMDXComponent=!0}}]);