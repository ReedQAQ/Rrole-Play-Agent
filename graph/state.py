from typing import Annotated,TypedDict,List,Literal
from typing_extensions import NotRequired
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from pydantic import BaseModel,Field

class AssistantState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

    summary:str
    
    check_result:NotRequired[dict[str, bool] | None]
    check_reason:NotRequired[str]
    retry_count:NotRequired[int]
    assistant_tool_rounds:NotRequired[int]
    memory_update_status:NotRequired[str]
    memory_task_id:NotRequired[str]



class CoreTraits(BaseModel):
    
    """目前只有大五人格"""
    openness:float
    conscientiousness:float
    extraversion:float
    agreeableness:float
    neuroticism:float


class SpeakingStyle(BaseModel):
    
    sentencelength:Literal["short","medium","long","mixed"]
    VocabulartLevel:Literal["academic","poem","casual","netword","mixed"]


class DynamicState(BaseModel):
    energy_level:int = Field(ge=0,le=100)
    mood:str


class PersonalityProfile(BaseModel):
    """完整人格画像"""
    core_traits:CoreTraits
    defense_mechanism:str = "未设定"
    speaking_style:SpeakingStyle
    dynamic_states:DynamicState
    interests: List[str]










class CharacterState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]
    summary:str
    
    check_result:NotRequired[dict[str, bool] | None]
    check_reason:NotRequired[str]
    retry_count:int#当前回答重新尝试次数

    character_id:str
    character_name:str
    character_identity:str
    personality:PersonalityProfile
    trigger_result:NotRequired[Literal["simple", "complex"]]
    cognitive_workspace:NotRequired[str]
    filtered_memories:NotRequired[str]
    memory_proposal:NotRequired[dict[str, object] | None]
    memory_update_status:NotRequired[str]
    memory_task_id:NotRequired[str]







