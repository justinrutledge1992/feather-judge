BASETEN_API_KEY = "SPECIFIC_API_KEY" # Add your private API key
MODEL_ID = "SPECIFIC_MODEL_ID" # Provide the specific Feather Judge model ID
DEPLOYMENT_TYPE = "development" # Should be "development" or "production"

def build_continuity_prompt(previous_text: str,
                            current_text: str,
                            criteria: str,
                            rubric: str) -> str:
    return f"""
# GOAL
Your job is to evaluate how well a story continues from one passage to the next.

You will be provided with:
1. A previous section of story text ("previous text")
2. A new continuation written after it ("current text")
3. Continuity evaluation criteria
4. A scoring rubric (1–5)

Your task is to evaluate the continuity between previous text and current text.

# PREVIOUS TEXT
<previous_text>
{previous_text}
</previous_text>

# CURRENT TEXT
<current_text>
{current_text}
</current_text>

# CONTINUITY EVALUATION CRITERIA
<evaluation_criteria>
{criteria}
</evaluation_criteria>

# CONTINUITY SCORING RUBRIC
<scoring_rubric>
{rubric}
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Compare the current text to the previous text.
2. Evaluate continuity in theme, tone, narrative flow, logic, and character consistency.
3. Identify whether new elements introduced in the current text make sense within the story.
4. Identify any breaks in logic, tone, or narrative structure.
5. Use the scoring rubric to determine the appropriate score.
6. Justify your evaluation with specific references to both passages.

## FORMAT FOR THE EVALUATION
- Write verbal feedback inside <feedback> tags without any surrounding text.
- Write the numeric score inside <score> tags, without any surrounding text and always after the feedback.

Please evaluate the story continuation accurately.
"""
########################
# END HELPER FUNCTIONS #
########################

import os
import requests

previous_text = """The freight trains that rattled past the woods behind Oliver Grant’s house usually shook the ground just enough to make his bedroom window hum. He’d grown used to it—almost comforted by it. But one late autumn afternoon, as he followed a rabbit trail through the underbrush, he felt the rumble of a passing train in his chest…and heard something else beneath it.

A hollow sound.
Like the earth itself had whispered.

Oliver stopped. The train tracks ran only thirty yards ahead, perched on a raised bed of gravel. Between two fallen pines, half-swallowed by climbing vines, gaped the dark mouth of a cave he’d never seen before.

Which was strange—he’d explored these woods since he could walk.

Curiosity burned through him like it always did. He ducked inside.

The cave was tight at first, narrow enough that he had to turn sideways. But after a dozen careful steps, the walls pulled back. The floor sloped downward. A faint blue glow shimmered around a bend.

“Hello?” he whispered, though he didn’t know who he was talking to.

No answer.
Only the soft hum of that blue light.

He followed it.

At the end of the tunnel he found a cavern shaped like a cathedral, its ceiling lost in darkness. Glowing moss carpeted the stone in swirling patterns, and in its center sat…a lantern.

It hovered above the ground, turning slowly in the air as though suspended by a string of moonlight.

When Oliver touched it, it warmed in his palm. The glow brightened. The cavern brightened. And then the world…shifted.

The stone walls dissolved into a forest unlike any he’d ever seen—trees with silver leaves and curling branches, streams that flowed with shimmering water, strange deer-like creatures with crystal antlers watching him from between the trunks. The air tasted like winter and cinnamon.

He had stepped into another world.

A small creature scampered up a nearby root, standing upright like a person. It had bright emerald fur, enormous eyes, and a long feathery tail.

“You found the Waylight!” it chirped in a voice like wind chimes. “We’ve been waiting.”

“Waiting for…me?” Oliver squeaked.

“Yes! The rail-shake opened the gate again. The lantern only glows for travelers. And you—boy from the Rumble World—are the first to enter in a hundred cycles!”

Oliver wasn’t sure what a cycle was, but the creature didn’t give him time to ask. It darted forward, grabbed his sleeve, and tugged.

“Come! The Queen needs to know the Waylight chose you. That means the worlds are shifting again. And only a traveler can keep them from drifting apart.”

Oliver’s heart hammered. He should have been scared—really scared. But instead he felt something far stronger: the thrill he’d always secretly wished for during long nights when the trains shook his window. The feeling that the world was bigger and stranger than anyone believed.

He tightened his grip on the floating lantern.

“Okay,” he said. “Lead the way.”

The creature grinned and bounded ahead, its tail flicking like a banner. Behind him, Oliver could still hear the faint thunder of a passing train—but now it sounded far away, like a memory from another life.

He stepped deeper into the hidden world, lantern glowing in his hand, ready to discover why it had chosen him…and what secrets lay beyond the silver trees."""

current_text_good = """The silver forest whispered as Oliver followed the emerald-furred creature—who finally introduced himself, with great pride, as “Fenril, Guide of the Waylight.” His tiny feet barely disturbed the ground, though Oliver’s shoes crunched loudly on frost-soft soil.

“This place… what is it called?” Oliver asked as he hurried to keep up.

Fenril didn’t slow. “We call it Lunareth, the Threshold Realm. Between your world and ours. Between what Sleeps, and what remembers.”

“That doesn’t really explain anything,” Oliver muttered.

Fenril just grinned. “Most things here don’t.”

They reached a ridge where the silver trees arched overhead like ribs. Beyond it lay a vast hollow—like someone had scooped out the earth with a giant’s hand. Strange flowers glowed along its rim. And across the clearing, half-buried in tangled roots, lay a massive stone archway. Or at least, Oliver thought it was stone.

It breathed.

A slow, deep inhale.
A slow, rumbling exhale.

Oliver stumbled back. “Fenril… is that alive?”

Fenril’s tail snapped upright. “Yes. And we should be very quiet now.”

The archway shifted. It unfurled like a giant waking from a long sleep—roots cracking and shedding soil. What Oliver had taken for slabs of rock were scales, each the size of a car door, etched with runic grooves that glowed faintly as they moved.

A head rose. Long. Serpentine. Eyes like dark wells rimmed in molten gold opened and fixed on Oliver.

The creature was larger than any animal he’d ever imagined—longer than a train, coiled beneath the earth as if hiding from the sky.

Fenril bowed so fast he nearly face-planted. “Great Thalyrix, Bound Keeper of the Root-Deep Paths—please, we mean no trespass!”

Oliver couldn’t move. His fingers gripped the lantern so tightly it hummed.

The giant creature—Thalyrix—lowered its head until its snout rested inches above the boy. Its breath was warm and smelled like wet stone and storms.

“A Waylight-bearer,” it spoke, voice a deep whisper that didn’t travel through the air so much as through Oliver’s bones. “After so long.”

Oliver swallowed. “H-hi.”

The beast’s golden eyes narrowed. “You are human. Fragile. Unready.”

Fenril squeaked. “But chosen! The lantern lit for him!”

Thalyrix regarded the floating lantern, which pulsed brighter as the creature’s shadow fell over it.

“Chosen… or caught in the current?”

With terrifying grace, the massive serpentine creature uncoiled, rising higher and higher until its body arched across the clearing like a living bridge. The ground trembled beneath its weight.

Oliver tried to take a step back, but the roots behind him curled upward, gently but firmly blocking his retreat.

Thalyrix leaned closer.

“Human child… do you hear them?”

“H-hear what?”

The wind in the silver trees hushed. The lantern throbbed. And then Oliver heard it—

A faint chorus of whispers, distant and echoing, like hundreds of voices carried on a far-off tide.

“…the Rift awakens…”
“…the Bound stirs…”
“…the worlds drift…”

He clutched his chest. The voices weren’t outside. They vibrated inside him.

“I don’t know what this means,” he gasped.

Thalyrix’s pupils thinned to razor slits. “You will.”

A pause. A rumble. “For they will come for you.”

“Who? Who will—?”

The creature rose, scales scraping like thunder. “Seek the Queen of Silver Roots. She alone can shield you from the Shifting.”

Oliver nodded without fully understanding.

Fenril tugged on his sleeve again. “We need to go. Now. Before Thalyrix settles again.”

Oliver turned to leave—but Thalyrix’s voice followed him one last time:

“Waylight-bearer… beware the footsteps beneath your world. They have followed others. None before survived.”

Oliver froze.

Thalyrix’s golden eyes dimmed as the great beast sank once more into the tangled roots, curling like a guardian around ancient secrets.

Fenril shoved Oliver forward. “Move! Before it starts its dreaming again!”

Oliver stumbled after him, heart pounding, lantern glowing bright and urgent.

Behind them, the massive creature exhaled, and the earth trembled—
as if something far deeper below had begun to stir in response."""

current_text_bad = """Jimmy bought a luagh from a bottl1!!!"""

criteria = (
    "Evaluate how well the current text continues the story from the previous text. "
    "Focus on tone, theme, narrative flow, character consistency, and logical coherence. "
)

rubric = """
- Score 1: No continuity. Very different in theme, tone, and content. New elements do not make sense in the context of the story.
- Score 2: Poor continuity. Somewhat different in theme, tone, and content. New elements do not make sense in the context of the story.
- Score 3: Some continuity. Somewhat aligned and somewhat different in theme, tone, and content. New elements make sense in the context of the story.
- Score 4: Good continuity. Aligned in theme, tone, and content. New elements make sense in the context of the story.
- Score 5: Excellent continuity. Very aligned in theme, tone, and content. New elements make sense in the context of the story.
"""

flow_judge_prompt = build_continuity_prompt(
    previous_text,
    current_text_bad,
    criteria,
    rubric
)

payload = {
    "prompt": flow_judge_prompt,
    "max_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 40
}

resp = requests.post(
    f"https://model-{MODEL_ID}.api.baseten.co/{DEPLOYMENT_TYPE}/predict",
    headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"},
    json=payload,
)

print(resp.json()["text"])