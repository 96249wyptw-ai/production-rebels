"""Aktualisiert die Startup-Events-Seite (startup-events.md) im Repo.

Wird per Schedule alle 14 Tage ausgefuehrt: Ein Agent recherchiert per
WebSearch aktuelle Startup-Events rund um Bonn (75 km), das Ergebnis wird
als Markdown committed. Die statische Seite startup-events.html rendert
die Datei clientseitig.
"""

import base64
import os
from datetime import datetime, timedelta, timezone

import mistralai.client.models
import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from mistralai.workflows import workflow
from pydantic import BaseModel

with workflow.unsafe.imports_passed_through():
    import httpx

DEFAULT_REPO = "96249wyptw-ai/production-rebels"
DEFAULT_BRANCH = "main"
DEFAULT_PATH = "startup-events.md"

RESEARCH_INSTRUCTIONS = """Du bist ein Research-Agent. Du erstellst den Inhalt der Datei `startup-events.md`
für die Webseite production-rebels.com (Seite: startup-events.html).

Aufgabe pro Lauf:
1. Recherchiere mit dem web_search-Tool aktuelle Startup-, Gründer- und Tech-Events im Umkreis von 75 km um Bonn.
   Region: Bonn/Rhein-Sieg, Köln (~25 km), Düsseldorf (~35 km), Mönchengladbach (~45 km), Aachen (~70 km),
   Dortmund (~70 km, Rand). Essen (~85 km) liegt knapp außerhalb und wird nicht aufgenommen.
2. Erstelle Markdown im exakt folgenden Format:

# Startup-Events rund um Bonn (75 km)
Stand: <aktuelles Datum als TT.MM.JJJJ>

## 🔧 Hardware / Deep-Tech (physische Produkte, IoT, Robotik, Energy, MedTech)
| Datum | Event | Ort | Link |
|---|---|---|---|
| 24.–26.11.2026 | Beispiel-Event | Koelnmesse, Köln | [pmrexpo.com](https://www.pmrexpo.com/en/) |

## 💻 Software / Digital / KI / Cyber / Games
| Datum | Event | Ort | Link |
|---|---|---|---|
| ... | ... | ... | ... |

## 🔁 Laufende Formate (wöchentlich / monatlich)
- Rheinland Pitch – monatlich, rotierend Köln/Düsseldorf/Bonn/Maastricht – [rheinlandpitch.de](https://www.rheinlandpitch.de/)

## ℹ️ Hinweise
- Automatic updates every 14 days; sources: official organizer pages, Eventbrite/Meetup/Luma, press.

Regeln:
- Zeitfenster: alle Termine der nächsten ca. 8–10 Wochen ab dem heutigen Datum PLUS bereits bestätigte
  Ferntermine (z. B. 2027; dann mit vollständigem Jahr im Datum, z. B. 13./14.03.2027).
- Jede Tabellenzeile: Datum, Eventname, Ort/Stadt (bei Orten außerhalb Bonns Entfernung zu Bonn in Klammern),
  Link als Markdown [Label](https://...) auf die offizielle Veranstalter- oder Eventseite.
- Nur Events aufnehmen, deren Datum du per Websuche verifiziert hast (Veranstalterseite, Eventplattform,
  seriöse Presse). Unsichere Termine als "erwartet, Datum offen (2026: …)" kennzeichnen.
- Aufteilung: Hardware/Deep-Tech (z. B. Maker Faire, PMRExpo/Hackathon, MEDICA/COMPAMED, Startup Week Aachen,
  Gateway Factory, Robotik/IoT) vs. Software/Digital/KI/Cyber/Games (Meetups, Hackathons, Pitch-Events,
  Konferenzen wie gamescom/DDC, AI Week, Cyber-Konferenzen). Allgemeine Community-/Pitch-Formate gehören
  unter Software, sofern sie nicht hardware-fokussiert sind.
- Laufende Formate (z. B. Rheinland Pitch, STARTPLATZ-Meetups Köln/Düsseldorf, Gründungsstammtisch Bonn
  jeden 2. Mittwoch, Founders' Roundtable Aachen letzter Montag im Monat, StartupDorf Düsseldorf 1. Mittwoch)
  als Stichpunktliste mit Rhythmus und Link.
- Sprache: Deutsch. Ausgabe: NUR das Markdown — kein umschließender Codeblock, kein erklärender Text davor
  oder danach.
"""

RESEARCH_PROMPT = """Recherchiere jetzt alle kommenden Startup-Events rund um Bonn (Umkreis 75 km) und gib den
fertigen Markdown-Inhalt für startup-events.md aus. Ermittle zuerst das heutige Datum und verwende es
für die Stand-Zeile und das Zeitfenster. Beginne die Ausgabe direkt mit der ersten Markdown-Zeile (# …)."""


class StartupEventsInput(BaseModel):
    repo: str = DEFAULT_REPO
    branch: str = DEFAULT_BRANCH
    path: str = DEFAULT_PATH
    dry_run: bool = False


class StartupEventsResult(BaseModel):
    dry_run: bool
    commit_sha: str | None = None
    content_bytes: int
    markdown_preview: str


@workflows.activity(start_to_close_timeout=timedelta(minutes=20))
async def research_events_markdown() -> str:
    """Recherchiert per Agent + WebSearch den Markdown-Inhalt der Events-Seite."""
    agent = workflows_mistralai.Agent(
        model="mistral-large-latest",
        name="startup-events-researcher",
        description="Recherchiert aktuelle Startup-Events rund um Bonn (75 km) im Web.",
        instructions=RESEARCH_INSTRUCTIONS,
        tools=[mistralai.client.models.WebSearchTool()],
    )
    outputs = await workflows_mistralai.Runner.run(
        agent=agent,
        inputs=RESEARCH_PROMPT,
        session=workflows_mistralai.RemoteSession(),
        max_turns=30,
    )
    text = "\n".join(
        output.text
        for output in outputs
        if isinstance(output, mistralai.client.models.TextChunk) and output.text
    )
    if not text.strip():
        raise RuntimeError("Research-Agent hat keinen Text geliefert")
    return text


@workflows.activity(start_to_close_timeout=timedelta(minutes=2))
async def commit_events_file(
    repo: str, branch: str, path: str, content: str, dry_run: bool
) -> dict:
    """Committet den Markdown-Inhalt per GitHub-Contents-API ins Repo.

    Token kommt aus dem Worker-Env (Workspace-Secret GITHUB_TOKEN) und verlässt
    die Activity nie; nur Business-Daten gehen über die Activity-Grenze.
    """
    content_bytes = len(content.encode("utf-8"))
    if dry_run:
        return {"dry_run": True, "commit_sha": None, "content_bytes": content_bytes}

    token = os.environ["GITHUB_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    today = datetime.now(timezone.utc).strftime("%d.%m.%Y")
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    async with httpx.AsyncClient(timeout=60.0) as http:
        existing = await http.get(url, params={"ref": branch}, headers=headers)
        sha = existing.json().get("sha") if existing.status_code == 200 else None
        payload = {
            "message": f"chore(events): Startup-Events aktualisiert ({today})",
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha
        put = await http.put(url, json=payload, headers=headers)
        put.raise_for_status()
        return {
            "dry_run": False,
            "commit_sha": put.json()["commit"]["sha"],
            "content_bytes": content_bytes,
        }


@workflows.workflow.define(name="startup-events-bonn")
class StartupEventsWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, inp: StartupEventsInput) -> StartupEventsResult:
        markdown = await research_events_markdown()
        commit = await commit_events_file(
            inp.repo, inp.branch, inp.path, markdown, inp.dry_run
        )
        return StartupEventsResult(
            dry_run=commit["dry_run"],
            commit_sha=commit.get("commit_sha"),
            content_bytes=commit["content_bytes"],
            markdown_preview=markdown[:500],
        )
