BASETEN_API_KEY = "" # Add your private API key
MODEL_ID = "" # Provide the specific Feather Judge model ID
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

current_text_bad = """Oliver quickly learned two things while jogging behind Fenril:

The little green fuzz-ball could move like he was being chased by a swarm of electrified bees.

Silver trees are extremely rude, because their branches like to poke you in the face for fun.

“Can we—wait—a second?” Oliver wheezed.

“No time!” Fenril squeaked, leaping over a log that Oliver absolutely did not leap over so much as trip over and slide down like a sad toboggan. “The Queen needs to see you!”

“All right,” Oliver muttered as he peeled moss off his forehead, “but if she asks why I look like I lost a wrestling match with a salad, I’m blaming you.”

They burst out onto a ridge overlooking an enormous hollow in the earth. It looked like someone had dropped a meteor, then politely tidied up the edges with landscaping tools.

And smack dab in the middle was what Oliver initially thought was a giant rock formation.

Then the rock exhaled.

Loudly.

Like a snoring walrus.

Oliver flinched so hard he almost threw the lantern into orbit.

“Fenril,” he whispered, “tell me that isn’t alive.”

Fenril puffed up proudly. “Oh, that most certainly is alive. That’s Thalyrix, Bound Keeper of the Root-Deep Paths!”

Thalyrix rumbled. The “rock” began to uncurl, sending trees tumbling down the side of its enormous body like startled squirrels.

Oliver yelped and hid behind Fenril, who was approximately towel-sized and therefore useless.

The creature finally lifted its massive head, blinking awake like someone who’d been rudely shaken during the best nap of their life. Two golden eyes the size of dinner plates focused on Oliver.

“WHO DISTURBS MY SLUMBER?” the creature boomed.

Fenril immediately sucked in a breath the way someone does before giving a long speech and announced:

“We do! But mostly him!”

He pointed directly at Oliver.

Oliver jabbed a finger at himself. “I—I—look, I didn’t even know you were here! This is a major miscommunication!”

Thalyrix lowered his enormous snout until Oliver could see his own terrified reflection in its scales.

“A human,” the beast intoned.
“Tiny.”
“Squishy.”
“Extremely edible.”

Oliver screamed like he’d swallowed a trumpet.

Fenril shook his head. “No no no, Great Thalyrix! Not edible! He’s holding the Waylight! That means he’s chosen!”

Thalyrix stared at the glowing lantern, leaning in until Oliver had no choice but to back up—straight into a tree root that rose suspiciously like a speed bump.

The lantern pulsed.

Thalyrix hummed thoughtfully.

“Ahhh. A Waylight-bearer.”
“Interesting.”
“Still edible.”

“Can we stop saying edible?” Oliver squeaked.

Thalyrix didn’t stop—he leaned in closer again until his golden eyes filled Oliver’s entire field of vision.

“DO YOU HEAR THE WHISPERS?”

Oliver blinked. “I… no? Should I?”

The lantern suddenly vibrated, and then:

whisper whisper whisper whisper

Oliver froze. “Okay, I hear them. Yep. Definitely hearing them. Are they supposed to sound like spooky gossip spread by ghosts who are very bad at whispering secretly?”

Fenril slapped both tiny hands over his muzzle, horrified. “Oliver! You don’t insult mystical cosmic warnings!”

Thalyrix snorted, blasting Oliver’s hair straight upward like he’d walked under a hand dryer.

“The Rift awakens,” Thalyrix boomed.
“The worlds drift.”
“Dark footsteps follow.”

“Footsteps?!” Oliver yelped. “From WHERE? And HOW BIG?”

Thalyrix raised one colossal brow ridge. “Bigger than you.”

“That could be anything,” Oliver protested. “Grasshoppers are bigger than me in this world!”

Thalyrix continued solemnly:
“Find the Queen of Silver Roots. She alone can protect you.”

Fenril tugged on Oliver’s sleeve. “And she’s less likely to eat you!”

“STOP SAYING THAT!” Oliver cried.

Thalyrix slowly settled back into the dirt like someone deflating a parade balloon.

“Go… quickly,” he rumbled. “Before I fall asleep and accidentally roll over.”

Oliver’s face went pale. “Roll over?”

Fenril screamed, “RUN!”

Oliver didn’t need to be told twice. He sprinted after Fenril while Thalyrix gave a final earth-shaking snore, curling back under the roots with the grace of a toppled mountain.

As Oliver stumbled into the trees, hair still standing straight up, Fenril squeaked:

“Well! That went better than expected!”

Oliver shot him a glare. “We have very different definitions of ‘better.’”

Behind them, something deep below the ground rumbled softly—like it was laughing."""

criteria = (
    "Evaluate how well the current text continues the story from the previous text. "
    "Focus on tone, theme, narrative flow, and logical coherence. "
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
    current_text_bad, # use this or current_text_bad for very different responses
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