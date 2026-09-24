"""Synthetic multi-turn conversation generator for a fictional phone
scheduling agent.

Domain: appointment scheduling for "Cedar Creek Home Services," a made-up
home repair business. No part of this module reads from or references any
real company, person, or call data.

Why multi-turn examples with short targets
--------------------------------------------
An earlier fine-tuning pass (see docs/lessons-from-five-iterations.md) used
single-turn synthetic pairs: one caller line in, one long agent reply out.
That shape taught the model to write a complete, well-organized paragraph
every time it spoke, because every training target was a complete
paragraph. On a live phone call that turned into monologuing: the model
would greet the caller, restate the request, ask a question, propose a
time, and summarize the appointment, all in one uninterrupted turn, because
that was the only turn shape it had ever been shown.

This generator instead produces one training example per agent turn inside
a real back-and-forth conversation, with the full system instruction and
conversation history as context and a single short agent turn (5 to 25
words, one question or one statement, never both) as the target. The model
learns to take one small step and then stop, the same way a human
receptionist does, because that is the only shape present in the data.

Determinism
-----------
``generate_dataset`` takes an explicit seed and does not read the system
clock, environment variables, or any external source. The same seed always
produces byte-identical output, which is what ``tests/test_prepare_data.py``
checks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

AGENT = "agent"
CALLER = "caller"

BUSINESS_NAME = "Cedar Creek Home Services"

SYSTEM_PROMPT = (
    f"You are the phone scheduling assistant for {BUSINESS_NAME}, a home "
    "repair and maintenance company. Handle one thing at a time. Each turn "
    "you take should be a single short question or a single short statement, "
    "never both. Do not invent availability, prices, or technician names "
    "that were not given to you. Keep every turn under 25 words."
)

FIRST_NAMES = [
    "Alex",
    "Jordan",
    "Sam",
    "Taylor",
    "Morgan",
    "Casey",
    "Riley",
    "Jamie",
    "Avery",
    "Quinn",
    "Reese",
    "Dana",
    "Drew",
    "Skyler",
    "Rowan",
    "Emerson",
    "Hayden",
    "Parker",
    "Elliot",
    "Charlie",
]

SERVICE_PHRASES = [
    "a leaking kitchen faucet",
    "the water heater not making hot water",
    "a tripped breaker in the garage",
    "the upstairs AC not cooling",
    "a running toilet in the hall bath",
    "installing a new ceiling fan",
    "an outlet that stopped working",
    "the furnace making a rattling noise",
    "a slow bathroom drain",
    "swapping out the garbage disposal",
    "a sump pump that won't shut off",
    "a thermostat that stopped responding",
]

DAY_PHRASES = [
    "today",
    "tomorrow",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "next Monday",
    "next Wednesday",
    "next Friday",
]

TIME_PHRASES = [
    "9 AM",
    "10:30 AM",
    "11 AM",
    "1 PM",
    "2:30 PM",
    "3 PM",
    "4:30 PM",
]

AREA_PHRASES = [
    "Maple Heights",
    "Brookline",
    "the Old Mill district",
    "Fairview Ridge",
    "Cedar Park",
    "Elm Grove",
]

TECH_NAMES = ["Marcus", "Priya", "Dan", "Lena", "Omar", "Wes"]


def _turns(*pairs: tuple[str, str]) -> list[tuple[str, str]]:
    return list(pairs)


def build_new_appointment(rng: random.Random) -> list[tuple[str, str]]:
    name = rng.choice(FIRST_NAMES)
    service = rng.choice(SERVICE_PHRASES)
    day = rng.choice(DAY_PHRASES)
    time = rng.choice(TIME_PHRASES)
    return _turns(
        (AGENT, f"Thanks for calling {BUSINESS_NAME}, how can I help you today?"),
        (CALLER, f"Hi, this is {name}. I need someone to look at {service}."),
        (AGENT, "What day this week works best for you?"),
        (CALLER, f"{day} would work well for me."),
        (AGENT, f"Do you prefer a morning or afternoon slot on {day}?"),
        (CALLER, f"Afternoon works better, maybe around {time}."),
        (AGENT, f"I have you down for {day} at {time} for {service}."),
        (CALLER, "That sounds good, thank you."),
        (AGENT, f"You're all set, the technician will call before arriving on {day}."),
    )


def build_reschedule(rng: random.Random) -> list[tuple[str, str]]:
    name = rng.choice(FIRST_NAMES)
    old_day = rng.choice(DAY_PHRASES)
    new_day = rng.choice([d for d in DAY_PHRASES if d != old_day])
    time = rng.choice(TIME_PHRASES)
    return _turns(
        (AGENT, f"Thanks for calling {BUSINESS_NAME}, how can I help you today?"),
        (CALLER, f"This is {name}, I need to move my appointment on {old_day}."),
        (AGENT, "What day would you like to move it to instead?"),
        (CALLER, f"Could we do {new_day} instead?"),
        (AGENT, f"What time on {new_day} works for you?"),
        (CALLER, f"Around {time} would be great."),
        (AGENT, f"I've moved your appointment to {new_day} at {time}."),
        (CALLER, "Perfect, thanks for the help."),
    )


def build_cancel(rng: random.Random) -> list[tuple[str, str]]:
    name = rng.choice(FIRST_NAMES)
    day = rng.choice(DAY_PHRASES)
    return _turns(
        (AGENT, f"Thanks for calling {BUSINESS_NAME}, how can I help you today?"),
        (CALLER, f"Hi, it's {name}, I need to cancel my visit on {day}."),
        (AGENT, "Can you confirm the name the appointment is under?"),
        (CALLER, f"Yes, it's under {name}."),
        (AGENT, f"Your appointment on {day} has been canceled."),
        (CALLER, "Great, thank you."),
        (AGENT, "Would you like to book a new time before we hang up?"),
        (CALLER, "Not right now, maybe later."),
        (AGENT, "No problem, call back anytime to rebook."),
    )


def build_availability_question(rng: random.Random) -> list[tuple[str, str]]:
    service = rng.choice(SERVICE_PHRASES)
    day = rng.choice(DAY_PHRASES)
    time = rng.choice(TIME_PHRASES)
    return _turns(
        (AGENT, f"Thanks for calling {BUSINESS_NAME}, how can I help you today?"),
        (CALLER, f"I'm wondering what your earliest opening is for {service}."),
        (AGENT, "Is this something you'd call urgent, or can it wait a few days?"),
        (CALLER, "It can wait a few days, no rush."),
        (AGENT, f"Our earliest non-urgent opening is {day} at {time}."),
        (CALLER, "That works for me, let's book it."),
        (AGENT, f"Great, you're confirmed for {day} at {time}."),
    )


def build_confirm_details(rng: random.Random) -> list[tuple[str, str]]:
    name = rng.choice(FIRST_NAMES)
    day = rng.choice(DAY_PHRASES)
    time = rng.choice(TIME_PHRASES)
    return _turns(
        (AGENT, f"Thanks for calling {BUSINESS_NAME}, how can I help you today?"),
        (CALLER, f"Hi, this is {name}, I wanted to confirm my appointment on {day}."),
        (AGENT, "Can you confirm the phone number on the account?"),
        (CALLER, "Sure, it's the number I'm calling from right now."),
        (AGENT, f"You're confirmed for {day} at {time}, see you then."),
        (CALLER, "Perfect, thanks for checking."),
    )


def build_service_area(rng: random.Random) -> list[tuple[str, str]]:
    area = rng.choice(AREA_PHRASES)
    served = rng.choice([True, False])
    turns = _turns(
        (AGENT, f"Thanks for calling {BUSINESS_NAME}, how can I help you today?"),
        (CALLER, f"Do you service homes out in {area}?"),
    )
    if served:
        turns += _turns(
            (AGENT, f"Yes, we cover {area} regularly."),
            (CALLER, "Great, I'd like to book a visit then."),
            (AGENT, "What kind of repair or install did you need?"),
        )
    else:
        turns += _turns(
            (AGENT, f"We don't currently route technicians out to {area}."),
            (CALLER, "That's too bad, thanks for checking anyway."),
            (AGENT, "You're welcome, sorry we couldn't help this time."),
        )
    return turns


def build_request_technician(rng: random.Random) -> list[tuple[str, str]]:
    tech = rng.choice(TECH_NAMES)
    day = rng.choice(DAY_PHRASES)
    time = rng.choice(TIME_PHRASES)
    return _turns(
        (AGENT, f"Thanks for calling {BUSINESS_NAME}, how can I help you today?"),
        (CALLER, f"Could I request {tech} specifically for my next visit?"),
        (AGENT, "Let me check who's scheduled that day before I confirm."),
        (CALLER, "Sure, take your time."),
        (AGENT, f"{tech} is available on {day} at {time}, does that work?"),
        (CALLER, "Yes, that works great."),
        (AGENT, f"You're booked with {tech} for {day} at {time}."),
    )


def build_emergency(rng: random.Random) -> list[tuple[str, str]]:
    name = rng.choice(FIRST_NAMES)
    time = rng.choice(TIME_PHRASES)
    return _turns(
        (AGENT, f"Thanks for calling {BUSINESS_NAME}, how can I help you today?"),
        (CALLER, f"This is {name}, I have water actively leaking under my sink."),
        (AGENT, "Is the water shutoff valve under the sink reachable right now?"),
        (CALLER, "Yes, I can get to it."),
        (AGENT, "Please turn that valve clockwise to stop the flow before we continue."),
        (CALLER, "Okay, done, the leak has stopped."),
        (AGENT, f"A technician can be there today by {time}."),
        (CALLER, "Thank you so much, that's a relief."),
        (AGENT, "You're welcome, hang tight until they arrive."),
    )


SCENARIOS = [
    build_new_appointment,
    build_reschedule,
    build_cancel,
    build_availability_question,
    build_confirm_details,
    build_service_area,
    build_request_technician,
    build_emergency,
]


@dataclass
class GeneratedDataset:
    train: list[dict] = field(default_factory=list)
    val: list[dict] = field(default_factory=list)


def _conversation_examples(turns: list[tuple[str, str]], seen_keys: set) -> list[dict]:
    """Turn one scripted conversation into one example per agent turn.

    Each example carries the full history before that agent turn. Examples
    whose (system, history) prompt already exists in ``seen_keys`` are
    dropped so the dataset never contains a duplicate prompt.
    """
    examples: list[dict] = []
    history: list[tuple[str, str]] = []
    for speaker, text in turns:
        if speaker == AGENT:
            key = (SYSTEM_PROMPT, tuple(history))
            if key not in seen_keys:
                seen_keys.add(key)
                examples.append(
                    {
                        "system": SYSTEM_PROMPT,
                        "history": [{"speaker": s, "text": t} for s, t in history],
                        "target": text,
                    }
                )
        history.append((speaker, text))
    return examples


def generate_dataset(
    num_examples: int = 300, val_fraction: float = 0.15, seed: int = 13
) -> GeneratedDataset:
    """Generate a synthetic multi-turn dataset, split at the conversation level.

    Splitting is done per conversation, not per example, so that a
    conversation's later turns (whose history includes its earlier turns)
    never end up in a different split than the earlier turns themselves.
    That would leak the same scripted content across train and val.

    Args:
        num_examples: target total number of agent-turn examples across
            both splits. The real total may differ slightly because
            conversations are kept whole; it will not be off by more than
            one conversation's worth of turns.
        val_fraction: approximate share of conversations routed to val.
        seed: RNG seed. Same seed, same output, always.
    """
    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = random.Random(seed)
    seen_keys: set = set()
    conversations: list[list[dict]] = []
    total = 0
    attempts = 0
    max_attempts = num_examples * 100

    while total < num_examples and attempts < max_attempts:
        attempts += 1
        builder = rng.choice(SCENARIOS)
        turns = builder(rng)
        examples = _conversation_examples(turns, seen_keys)
        if not examples:
            continue
        conversations.append(examples)
        total += len(examples)

    rng.shuffle(conversations)

    train: list[dict] = []
    val: list[dict] = []
    val_target = max(1, round(len(conversations) * val_fraction))
    for i, conv in enumerate(conversations):
        if i < val_target:
            val.extend(conv)
        else:
            train.extend(conv)

    rng.shuffle(train)
    rng.shuffle(val)

    for i, ex in enumerate(train):
        ex["id"] = f"train-{i:04d}"
    for i, ex in enumerate(val):
        ex["id"] = f"val-{i:04d}"

    # Move id to the front of each dict for a nicer JSONL read.
    train = [_reorder(ex) for ex in train]
    val = [_reorder(ex) for ex in val]

    return GeneratedDataset(train=train, val=val)


def _reorder(example: dict) -> dict:
    return {
        "id": example["id"],
        "system": example["system"],
        "history": example["history"],
        "target": example["target"],
    }
