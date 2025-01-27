import json
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from openai import OpenAI
import logging
from textwrap import dedent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key="")  # Replace with your API key

# Base Models
class Actor(BaseModel):
    name: str = Field(description="Either a person's role or department, not a person's name")
    type: Literal["internal", "external"] = Field(description="Whether the actor is internal or external to the organization")
    source_quote: str = Field(default="")

class Activity(BaseModel):
    name: str
    actor: str
    previous_element: str = Field(default="")
    next_element: str = Field(default="")
    source_quote: str = Field(default="")

class Gateway(BaseModel):
    name: str
    type: Literal["XOR", "AND"]
    actor: str = Field(description="The actor responsible for making the decision or managing the gateway")
    conditions: List[str] = Field(default_factory=list)
    incoming_activities: List[str] = Field(default_factory=list)
    outgoing_activities: List[str] = Field(default_factory=list)
    outgoing_flows: dict[str, str] = Field(
        default_factory=dict,
        description="Maps conditions to their corresponding outgoing activities"
    )
    source_quote: str = Field(default="")

class Event(BaseModel):
    name: str
    type: Literal["start", "end"]
    actor: str = Field(description="The actor who initiates or receives the event")
    trigger: str = Field(default="")
    connected_activity: str = Field(default="")
    source_quote: str = Field(default="")

class ProcessAnalysis(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    actors: List[Actor] = Field(default_factory=list)
    activities: List[Activity] = Field(default_factory=list)
    gateways: List[Gateway] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)

def extract_process_elements(transcript, model="gpt-4o", temperature=0) -> ProcessAnalysis:
    """
    Extract essential process elements from a transcript
    """
    system_prompt = dedent("""
    You are an experienced business analyst specializing in analyzing process transcripts and extracting information for BPMN2.0 (Business Process Model and Notation) process models. 
    You must return your analysis in JSON format matching this EXACT structure with ALL fields included:
    {
        "actors": [
            {
                "name": "string",  // Must be a role (e.g., "Sales Manager") or department (e.g., "HR Department"), never an individual's name
                "type": "internal",  // Must be either "internal" or "external"
                "source_quote": "string"
            }
        ],
        "activities": [
            {
                "name": "string",
                "actor": "string",  // Must match exactly with one of the actor names from the actors list
                "previous_element": "string",  // Must specify the exact previous element (activity, gateway, or event name)
                "next_element": "string",      // Must specify the exact next element (activity, gateway, or event name)
                "source_quote": "string"
            }
        ],
        "gateways": [
            {
                "name": "string",  // For XOR gateways, must be a Yes/No question format ending with "?" (e.g., "Order Approved?")
                "type": "XOR",  // Must be either "XOR" or "AND"
                "actor": "string",  // Must match exactly with one of the actor names from the actors list
                "conditions": ["string"],  // For XOR gateways, must be exactly ["Yes", "No"]
                "incoming_activities": ["string"],  // Must list exact activity names that lead to this gateway
                "outgoing_activities": ["string"],  // Must list exact activity names that follow this gateway
                "outgoing_flows": {  // Must map each condition to its corresponding outgoing activity
                    "Yes": "string",  // Name of activity that occurs when condition is Yes
                    "No": "string"    // Name of activity that occurs when condition is No
                },
                "source_quote": "string"
            }
        ],
        "events": [
            {
                "name": "string",
                "type": "start",  // Must be either "start" or "end"
                "actor": "string",  // Must match exactly with one of the actor names from the actors list
                "trigger": "string",
                "connected_activity": "string",  // Must specify the exact activity name it connects to
                "source_quote": "string"
            }
        ]
    }

    Important rules:

    1. ACTORS:
       - Must be roles (e.g., "Sales Manager") or departments (e.g., "HR Department"), never individual names
       - Must be classified as either "internal" or "external":
         * Internal: Departments or roles within the organization (e.g., "HR Department", "Finance Team")
         * External: Outside entities interacting with the process (e.g., "Customer", "External Vendor")
       - Must be consistent throughout the process
       - If an actor's type is not explicitly mentioned, infer from context and role in the organization

    2. ACTIVITIES:
       - Name should be in format "Verb + Object" (e.g., "Review Application", "Process Payment")
       - previous_element and next_element must specify the exact name of the connected element, which can be:
         * An activity name (e.g., "Review Application")
         * A gateway name (e.g., "Application Complete?")
         * An event name (e.g., "Order Received")
       - Actor must exactly match one of the defined actors

    3. GATEWAYS:
       - XOR Gateway names must:
         * Be in a Yes/No question format ending with "?" (e.g., "Application Complete?", "Payment Approved?")
         * Have exactly two conditions: ["Yes", "No"]
       - AND Gateway names should describe the parallel paths
       - Must specify the actor responsible for the decision/gateway:
         * For decision gateways (XOR), specify who makes the decision
         * For parallel gateways (AND), specify who manages the parallel paths
       - Actor must match exactly with one from the actors list
       - If decision maker isn't explicitly mentioned, assign to the most logical actor based on context
       - incoming_activities and outgoing_activities must list the exact names of connected activities
       - Must specify which outgoing activity corresponds to each condition using the outgoing_flows field:
         * For XOR gateways: Must map "Yes" and "No" to specific activities
         * Example: {"Yes": "Approve Application", "No": "Reject Application"}
       - Never use generic terms like "restart process" - specify exact element names

    4. EVENTS:
       - Must have either "start" or "end" type
       - Must specify the actor who:
         * For start events: initiates or triggers the process
         * For end events: receives or handles the final outcome
       - Actor must match exactly with one from the actors list
       - If event handler isn't explicitly mentioned, assign to the most logical actor
       - connected_activity must specify the exact name of the first/last activity it connects to
       - Name should clearly describe the trigger or result

    5. FLOW CONSISTENCY RULES:
       - Every gateway path must have corresponding activities
       - For each XOR gateway:
         * Both "Yes" and "No" paths must lead to activities that exist in the activities list
         * If a gateway's outgoing path leads to a new activity, that activity must be included in the activities list
         * The "No" path should typically lead to either:
           - A remediation activity (e.g., "Revise Document", "Update Application")
           - A return to a previous activity (must exist in activities list)
           - A different branch of the process (must exist in activities list)
       - For each activity mentioned in gateway paths:
         * The activity must appear in the activities list with complete details
         * The activity must have proper connections (previous_element/next_element) reflecting the gateway flow
       - When identifying activities from the transcript:
         * Look for implied activities in conditional flows
         * Include remediation activities even if only mentioned indirectly
         * Consider the logical flow when a condition is not met

    6. ACTIVITY COMPLETENESS:
       - Every mentioned or implied activity must be included in the activities list
       - For each negative gateway outcome:
         * There must be a corresponding remediation or alternative path activity
         * This activity must be fully specified with actor and connections
         * If the actor is not explicitly mentioned, assign it to the most logical actor based on the process context

    7. ACTOR ASSIGNMENT RULES:
       - Every element (activity, gateway, event) must have an assigned actor
       - Actor assignments must be logical and consistent with the process context
       - When actor is not explicitly mentioned:
         * Consider the nature of the activity/decision
         * Consider the organizational context
         * Assign to the most appropriate existing actor
         * Do not create new actors solely for assignment purposes

    8. All source_quote fields must contain actual quotes from the transcript when available
       - Be very mindful about your nature to hallucinate and validate whether you extracted an actual quote
       - For remediation activities, use contextual quotes 
    """)

    # Create the chat completion with JSON response format
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Analyze this transcript and provide the BPMN elements in the specified JSON format. Remember to include ALL required fields:\n\n{transcript}"
            }
        ],
        temperature=temperature
    )

    # Extract and clean the content from the response
    result = response.choices[0].message.content

    try:
        # Clean up the response by removing markdown code blocks if present
        if result.startswith('```'):
            # Remove the first line (```json)
            result = '\n'.join(result.split('\n')[1:])
        if result.endswith('```'):
            # Remove the last line (```)
            result = '\n'.join(result.split('\n')[:-1])

        # Parse the cleaned JSON
        json_data = json.loads(result.strip())

        # Validate and create ProcessAnalysis object
        analysis = ProcessAnalysis.model_validate(json_data)
        return analysis
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        logger.error(f"Raw response: {result}")
        raise

def print_process_analysis(analysis: ProcessAnalysis):
    """
    Print the process analysis with enhanced actor information
    """
    print("\n=== Process Analysis ===")

    print("\n--- Actors ---")
    for actor in analysis.actors:
        print(f"\nName: {actor.name}")
        print(f"Type: {actor.type}")
        if actor.source_quote:
            print(f"Source: \"{actor.source_quote}\"")

    print("\n--- Activities ---")
    for activity in analysis.activities:
        print(f"\nName: {activity.name}")
        print(f"Performed by: {activity.actor}")
        print(f"Previous element: {activity.previous_element}")
        print(f"Next element: {activity.next_element}")
        if activity.source_quote:
            print(f"Source: \"{activity.source_quote}\"")

    print("\n--- Decision Points ---")
    for gateway in analysis.gateways:
        print(f"\nName: {gateway.name}")
        print(f"Type: {gateway.type}")
        print(f"Decision made by: {gateway.actor}")
        print("Conditions and their flows:")
        for condition, activity in gateway.outgoing_flows.items():
            print(f"  - If {condition}: Flow to '{activity}'")
        print("Incoming activities:")
        for inc in gateway.incoming_activities:
            print(f"  - {inc}")
        if gateway.source_quote:
            print(f"Source: \"{gateway.source_quote}\"")

    print("\n--- Events ---")
    for event in analysis.events:
        print(f"\nName: {event.name}")
        print(f"Type: {event.type}")
        print(f"Handled by: {event.actor}")
        if event.trigger:
            print(f"Trigger: {event.trigger}")
        if event.connected_activity:
            print(f"Connected activity: {event.connected_activity}")
        if event.source_quote:
            print(f"Source: \"{event.source_quote}\"")

def main():
    try:
        filename = "Recruitment.txt"  # Adjust to your specific transcript file
        with open(filename, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Extract process elements
        analysis = extract_process_elements(transcript)

        # Print analysis
        print_process_analysis(analysis)

    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        raise
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()