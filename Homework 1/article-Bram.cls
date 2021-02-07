\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{article-Bram}[2019/07/01 v1.1 article-Bram]

% new package options
\newif\if@print
\DeclareOption{print}{\@printtrue}
\newif\if@serif
\DeclareOption{serif}{\@seriftrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax
\LoadClass[a4paper, twocolumn, twoside]{article}

% Overall document layout
\RequirePackage[top=1in, bottom=1in, left=0.6in, right=0.6in, twoside]{geometry} % page setup
\RequirePackage[small]{titlesec} % titles layout
\setcounter{tocdepth}{3}
\RequirePackage{graphicx}\graphicspath{{./Figures/}}
\RequirePackage{parskip} % no indents and better paragraph rules
\usepackage{scrextend}\deffootnote[1em]{1em}{1em}{\textsuperscript{\thefootnotemark}\,} % footnote settings
\RequirePackage{fancyhdr} % fancy page layout, with header rule
\fancyhead[RO,LE]{\thepage}
\fancyhead[RE]{\leftmark}
\fancyhead[LO]{\rightmark}
\pagestyle{fancy}
\fancyfoot[c]{\thepage}

% Fonts
\RequirePackage{amsmath} % needs before math font selection
\RequirePackage{fontspec} % advanced font selector
\setmainfont[Scale = MatchLowercase]{DejaVu Serif} % open serif font
\setsansfont[Scale = MatchLowercase]{DejaVu Sans} % open sans font
\setmonofont[Scale = MatchLowercase]{DejaVu Sans Mono}
\if@serif
	\usepackage{unicode-math}
	\setmathfont[Scale = MatchLowercase]{DejaVu Math TeX Gyre} % math font equal in style and size to DejaVu Serif
	\SetSymbolFont{symbols}{bold}{OMS}{zavm}{m}{n}
\else
	\renewcommand{\familydefault}{\sfdefault}
	\RequirePackage{arevmath} % math font that is equal in style DejaVu Sans, but better a, nu, omega, l, x ect
	\DeclareMathSizes{8}{6.32}{5}{5}
	\DeclareMathSizes{9}{7.1}{6}{5}
	\DeclareMathSizes{10}{7.9}{6}{5}
	\DeclareMathSizes{10.95}{8.65}{7}{5}
	\DeclareMathSizes{12}{9.4}{7.6}{6}
	\SetSymbolFont{symbols}{bold}{OMS}{zavm}{m}{n} % surpress warning of bold math sybols
\fi
\RequirePackage{bm} % bold greek and other symbols in math mode
\RequirePackage{csquotes} % proper quotation marks
\RequirePackage{physics} % propper fonts for derivatives, operators ect.

% Packages for extra enviroments/options
\RequirePackage[usenames, svgnames, table]{xcolor} % color selection by name
\RequirePackage[justification=centering]{caption} % caption settings
\RequirePackage{stfloats} % full size figures on the correct page
\RequirePackage[shortlabels]{enumitem} % list/items with advanced layout options
\setlist{nosep} % list/items with reduced white space
\RequirePackage{booktabs} % more table settings
\RequirePackage{multirow} % tables with multirow cells
\RequirePackage[list=true]{subcaption} % subcaption settings
\DeclareCaptionSubType[alph]{figure} % subfigure caption alphabetic
\captionsetup[subfigure]{labelformat=brace,labelsep=space} % subcaption format
\RequirePackage{todonotes}

% Internal referencing and external links
\RequirePackage[hidelinks]{hyperref} % hyperrefs without ugly borders
\urlstyle{same} % urls in standard font
\if@print
	\hypersetup{colorlinks=false} % color options for all links
\else
	\hypersetup{colorlinks=true, urlcolor=MediumBlue, filecolor=MediumBlue, linkcolor=Crimson, citecolor=Green} % color options for links
\fi
\RequirePackage{cleveref} % smarter referencing to internal objects

% Math vector and matrices upright and bold
\def\vec#1{\ensuremath{\mathchoice
		{\mbox{\boldmath$\displaystyle\mathbf{#1}$}}
		{\mbox{\boldmath$\textstyle\mathbf{#1}$}}
		{\mbox{\boldmath$\scriptstyle\mathbf{#1}$}}
		{\mbox{\boldmath$\scriptscriptstyle\mathbf{#1}$}}}}

% Math exponetial as operator e^
\let\exp\relax\DeclareMathOperator{\exp}{e}

% New commands
\renewcommand*\title[1]{\def\@title{#1}\hypersetup{pdftitle=#1}}
\renewcommand*\author[1]{\def\@author{#1}\hypersetup{pdfauthor=#1}}
\newcommand*\keywords[1]{\def\@keywords{#1}\hypersetup{pdfkeywords={#1}}}
\newcommand*\institute[1]{\def\@institute{#1}}

% Change \maketitle behavior
\def\@maketitle{%
	\newpage
	\null
	\vskip 2em%
	\begin{center}%
		\let \footnote \thanks
		{\huge \@title \par}%
		\vskip 1.5em%
		{\large
			\lineskip .5em%
			\begin{tabular}[t]{c}%
				\Large \@author
			\end{tabular}\par}%
		\ifdefined\@institute \@institute \fi%
		\vskip 0.5em%
		{\@date}%
	\end{center}%
	\par
	\vskip 1.5em}

% setup one column abstract
\RequirePackage{environ}
\RequirePackage{abstract}

\RenewEnviron{abstract}{%
	\twocolumn[%
		\maketitle%
		\rule{\textwidth}{0.4pt}
		\vspace{-2\baselineskip}
		\section*{Abstract}%
			\BODY%

		\ifdefined\@keywords
			\vspace{\baselineskip}
			Keywords: \@keywords\\%
		\fi
		\rule{\textwidth}{0.4pt}\\
	]\saythanks%
}

% References
\RequirePackage[style=ieee, citestyle=numeric-comp, backend=biber]{biblatex} % IEEEtran cite
\addbibresource{bibliography.bib} % name and location of bibliography libary
\setlength\bibitemsep{0.3\baselineskip} % extra white space

\if@print
	% Normal IEEEtran style with doi printed
\else
	% Title as link to DOI reference
	\ExecuteBibliographyOptions{doi=false}
	\newbibmacro{string+doi}[1]{%
	\iffieldundef{doi}{#1}{\href{http://dx.doi.org/\thefield{doi}}{#1}}}
	\DeclareFieldFormat{title}{\usebibmacro{string+doi}{\mkbibemph{#1}}}
	\DeclareFieldFormat[article]{title}{\usebibmacro{string+doi}{\mkbibquote{#1}}}
\fi