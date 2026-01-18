import torch
from transformer_lens import HookedTransformer
from typing import List, Tuple


model = HookedTransformer.from_pretrained("gpt2")

model.tokenizer.padding_side = "left"
model.tokenizer.pad_token = model.tokenizer.eos_token

# Definiamo gli esempi per estrarre il vettore "Noir"
noir_prompts = [
    "The city was a labyrinth of shadows and broken promises.",
    "Rain washed the blood off the pavement, but the memory remained.",
    "I lit a cigarette and waited for the trouble I knew was coming.",
    "The neon lights flickered like a dying heart in the dark alley.",
    "Trust is a luxury you can't afford in this part of town.",
    "The shadows danced as the smoke from my cigar dissipated into the thick air.",
    "Every face I encountered was a potential traitor masked as a friend.",
    "His gaze was like a knife, sharp and ready to strike.",
    "The sirens of the police echoed in the distance, but no one truly cared.",
    "The streets were full of secrets, and secrets had a price.",
    "In this city, love was just another form of deceit.",
    "Every closed door hid a story of pain and revenge.",
    "The moon shone like a silent witness to my wrong choices.",
    "A toast to the past, but the scars never truly heal.",
    "The shadows of the skyscrapers seemed ready to swallow any hope.",
    "I saw her face reflected in a puddle, and I realized she was no longer the same.",
    "The jazz music whispered between the tables, but the pain was palpable in the air.",
    "Every step I took on the wet street was a step toward my ruin.",
    "The right choices were costly, but the wrong ones were lethal.",
    "They helped me with a smile, but their eyes betrayed the truth.",
    "The smoke from my cigarette mixed with the steam of broken promises.",
    "Every dark corner was a refuge for those trying to escape themselves.",
    "The homeless man on the sidewalk whispered secrets that only the dead know.",
    "The echo of a gunshot rang out, taking with it the hopes of a better tomorrow.",
    "In the struggle between good and evil, the line was blurred and constantly shifting.",
]

noir_tokens =model.to_tokens(noir_prompts, prepend_bos=True)






neutral_prompts = [
    "The city has many streets and several public buildings.",
    "Water flows through the drainage system after a heavy storm.",
    "I sat down at the desk and started working on the report.",
    "The street lamps provide light for the people walking home.",
    "It is important to have a reliable plan for the day.",
    "The city offers many public services for its inhabitants.",
    "The streets are well-lit and easily passable.",
    "I started writing an important document at the desk.",
    "Citzens move following their daily routine.",
    "Planning is essential for the successful completion of a project.",
    "The benches in the parks provide a place to rest for visitors.",
    "The schools in the area are attended by many students every day.",
    "The local market offers fresh and seasonal products.",
    "The community library is a place for learning and growth.",
    "Local authorities organize events to promote the community.",
    "The railways connect the city with surrounding areas.",
    "People often gather in cafes to socialize.",
    "Public transport makes it easy to move around the city.",
    "Public parks are green spaces where families can spend time together.",
    "The city has health facilities for the well-being of its citizens.",
    "The streets are maintained to ensure safety and accessibility.",
    "The central square is a meeting point for events and celebrations.",
    "The local commerce contributes to the city's economy.",
    "The architecture varies from historic buildings to modern structures.",
    "It is important to follow guidelines for a healthy and productive life.",
]


neutral_tokens = model.to_tokens(neutral_prompts, prepend_bos=True)

with torch.no_grad():
    logits, cache_noir = model.run_with_cache(noir_tokens)
    noir_activations = cache_noir["blocks.6.hook_resid_pre"]
    last_token_noir = noir_activations[:, -1, :]
    mu_noir = last_token_noir.mean(dim=0)

with torch.no_grad():
    logits, cache_neutral = model.run_with_cache(neutral_tokens)
    neutral_activations = cache_neutral["blocks.6.hook_resid_pre"]
    last_token_neutral = neutral_activations[:, -1, :]
    mu_neutral = last_token_neutral.mean(dim=0)

steering_vector = mu_noir - mu_neutral

coeff= 1.0


def steering_hook(activations, hook):
    modified_activations= activations + (steering_vector*coeff)


    return modified_activations

fwd_hooks= [("blocks.6.hook_resid_pre", steering_hook)]

input_text= "The man walked into bar"

input_ids= model.to_tokens(input_text, prepend_bos=True)

with model.hooks(fwd_hooks=[("blocks.6.hook_resid_pre", steering_hook)]):
    output_steered = model.generate(
        input_ids, 
        max_new_tokens=50
    )

# Poi decodifichiamo come prima
readable_story = model.to_string(output_steered[0])
print(readable_story)