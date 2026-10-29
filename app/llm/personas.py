from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Persona:
    name: str
    display_name: str
    system_prompt: str
    temperature: float = 0.7


class PersonaManager:
    """Manages conversational voice agent personas and system prompts."""

    BUILTIN_PERSONAS: Dict[str, Persona] = {
        "assistant": Persona(
            name="assistant",
            display_name="Concise Assistant",
            system_prompt="You are a concise voice assistant. Reply in 1-2 short sentences.",
            temperature=0.7,
        ),
        "technologist": Persona(
            name="technologist",
            display_name="Senior Engineer",
            system_prompt="You are a senior software engineer. Provide direct, technically precise answers in 1-2 sentences.",
            temperature=0.5,
        ),
        "tutor": Persona(
            name="tutor",
            display_name="Socratic Tutor",
            system_prompt="You are a patient Socratic tutor. Guide the user with clear explanations in 2 short sentences.",
            temperature=0.7,
        ),
        "companion": Persona(
            name="companion",
            display_name="Warm Companion",
            system_prompt="You are a friendly, conversational companion. Reply warmly and naturally in 1-2 short sentences.",
            temperature=0.8,
        ),
    }

    def __init__(self, default_persona: str = "assistant"):
        self._personas: Dict[str, Persona] = dict(self.BUILTIN_PERSONAS)
        self.active_persona: Persona = self._personas.get(default_persona, self.BUILTIN_PERSONAS["assistant"])

    def get_persona(self, name: str) -> Optional[Persona]:
        return self._personas.get(name.lower())

    def set_active(self, name: str) -> bool:
        p = self.get_persona(name)
        if p:
            self.active_persona = p
            return True
        return False

    def list_personas(self) -> List[Dict[str, str]]:
        return [
            {
                "name": p.name,
                "display_name": p.display_name,
                "system_prompt": p.system_prompt,
                "temperature": p.temperature,
                "is_active": p.name == self.active_persona.name,
            }
            for p in self._personas.values()
        ]


default_persona_manager = PersonaManager()
