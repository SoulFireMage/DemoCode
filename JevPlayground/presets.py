"""Starter content. Each candidate: first line is the name, the rest is the description.

Summaries are deliberately short and neutral; sources are the usual entry points
into each literature. Edit freely - the point is that the model judges what is
written in the boxes, so what you put there shapes the answers.
"""

CONSCIOUSNESS_CONTEXT = """\
Evidence notes (edit or extend):
- The COGITATE adversarial collaboration (Cogitate Consortium, Nature, 2025) pre-registered \
predictions from Integrated Information Theory and Global Neuronal Workspace Theory. Results \
challenged key predictions of both: no sustained synchronisation within posterior cortex (an IIT \
prediction) and no clear prefrontal 'ignition' at stimulus offset (a GNWT prediction).
- General anaesthesia (e.g. propofol) disrupts long-range cortical integration and \
fronto-parietal feedback connectivity.
- Blindsight: patients with V1 damage can discriminate stimuli they report not seeing.
- Review: Seth & Bayne, 'Theories of consciousness', Nature Reviews Neuroscience (2022).
"""

CONSCIOUSNESS_THEORIES = [
    """Global Workspace Theory (GWT / GNWT)
Consciousness is global availability: information becomes conscious when it wins competition \
for a limited-capacity 'workspace' and is broadcast to many specialised processors. The neuronal \
version proposes non-linear 'ignition' in fronto-parietal networks.
Sources: Baars, A Cognitive Theory of Consciousness (1988); Dehaene & Naccache, Cognition (2001); \
Dehaene, Consciousness and the Brain (2014).""",
    """Integrated Information Theory (IIT)
Starts from axioms about experience and posits that consciousness is identical to a system's \
maximally irreducible integrated cause-effect structure, quantified as phi. Predicts the posterior \
cortical 'hot zone' matters more than prefrontal cortex, and that some simple systems have \
non-zero consciousness.
Sources: Tononi, BMC Neuroscience (2004); Oizumi, Albantakis & Tononi, PLoS Comp Biol (2014); \
Albantakis et al., IIT 4.0, PLoS Comp Biol (2023).""",
    """Higher-Order Theories (HOT / HOROR / PRM)
A mental state is conscious when it is the target of a suitable higher-order representation - \
a thought or meta-representation that one is in that state. Prefrontal metacognitive circuits \
are often proposed as the substrate.
Sources: Rosenthal, Consciousness and Mind (2005); Lau & Rosenthal, Trends Cogn Sci (2011); \
Brown, Lau & LeDoux, Trends Cogn Sci (2019).""",
    """Recurrent Processing Theory (RPT)
Feedforward sweeps through sensory cortex are unconscious; local recurrent (feedback) processing \
within sensory areas is sufficient for phenomenal experience, even without report or global \
access.
Sources: Lamme & Roelfsema, Trends Neurosci (2000); Lamme, Trends Cogn Sci (2006).""",
    """Predictive Processing / 'Beast Machine'
The brain is a prediction engine; conscious contents are the brain's best guesses about the \
causes of its sensory signals. Selfhood and emotion arise from interoceptive prediction tied to \
regulating the living body.
Sources: Clark, Surfing Uncertainty (2016); Seth, Being You (2021); Seth & Hohwy, Cognitive \
Neuroscience (2021).""",
    """Attention Schema Theory (AST)
The brain builds a simplified internal model of its own attention. Claims of subjective awareness \
are what that schematic model reports; awareness is the brain's description of attention, not an \
extra property.
Sources: Graziano, Consciousness and the Social Brain (2013); Graziano & Webb, Frontiers in \
Psychology (2015).""",
    """Orchestrated Objective Reduction (Orch-OR)
Consciousness arises from quantum computations in neuronal microtubules that terminate by an \
objective, gravity-related collapse of the wavefunction, proposed by Penrose.
Sources: Penrose, The Emperor's New Mind (1989); Hameroff & Penrose, Physics of Life Reviews (2014).""",
    """Illusionism
Phenomenal consciousness, as philosophers usually conceive it, does not exist; what needs \
explaining is why we are so strongly disposed to believe we have ineffable qualia (the \
'illusion problem').
Sources: Dennett, Consciousness Explained (1991); Frankish, Journal of Consciousness Studies (2016).""",
    """Panpsychism (Russellian monism)
Consciousness, or proto-consciousness, is a fundamental feature of matter; physics describes only \
the structure of matter, and experience is its intrinsic nature. Faces the 'combination problem'.
Sources: Chalmers, The Conscious Mind (1996); Goff, Galileo's Error (2019).""",
]

# Optional leading "weight |" sets that criterion's weight in the composite.
CONSCIOUSNESS_CRITERIA = """\
2 | Does `candidate` make at least one specific empirical prediction that a neuroscience experiment could show to be false?
2 | Is `candidate` consistent with all of the findings listed in `shared_context`?
1 | Does `candidate` attempt to explain why experience feels like something from the inside, rather than only which processes are conscious?
1 | Does `candidate` explain the findings about general anaesthesia listed in `shared_context`?
0 | Is `candidate` widely regarded by consciousness researchers as one of the leading scientific theories?"""

CONSCIOUSNESS_PICK = "Based on the descriptions and `shared_context`, which candidate theory is most likely to be correct?"

SCORE_LEVELS_DEFAULT = """\
Not at all
Weakly
Partly
Largely
Fully"""

PRESETS = {
    "Theories of consciousness": dict(
        context=CONSCIOUSNESS_CONTEXT,
        candidates=CONSCIOUSNESS_THEORIES,
        criteria=CONSCIOUSNESS_CRITERIA,
        pick=CONSCIOUSNESS_PICK,
    ),
    "Support tickets (sanity check)": dict(
        context="Our refund policy: duplicate charges are refunded within 5 days.",
        candidates=[
            "Ticket 1\nHelp! I was charged twice for order A-104 and rent is due tomorrow.",
            "Ticket 2\nHow do I change the email address on my account?",
            "Ticket 3\nYour app crashes every time I open the settings page.",
        ],
        criteria="1 | Does `candidate` express urgency?\n1 | Is `candidate` about money or billing?\n"
                 "1 | Does `shared_context` say what will happen for the issue in `candidate`?",
        pick="Which candidate ticket should a human handle first?",
    ),
}
