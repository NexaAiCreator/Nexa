"""
Agent planning and execution service for Nexa Browser autonomy.
Handles goal parsing, action planning, workflow generation, and result synthesis.
"""

import json
import re
from typing import Any, Optional, List, Dict
from enum import Enum
from datetime import datetime


class ActionType(str, Enum):
    """Action types for browser automation"""
    CLICK = "click"
    FILL = "fill"
    READ = "read"
    NAVIGATE = "navigate"
    WAIT = "wait"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"


class StepType(str, Enum):
    """Step types for workflow"""
    ACTION = "action"
    DECISION = "decision"
    CLARIFICATION = "clarification"
    SYNTHESIZE = "synthesize"


class RiskLevel(str, Enum):
    """Risk levels for permission analysis"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionPlan:
    """Represents a single action to execute"""
    
    def __init__(self, 
                 action_id: str,
                 action_type: ActionType,
                 params: Dict[str, Any],
                 selector: Optional[str] = None,
                 required_permissions: Optional[List[str]] = None):
        self.action_id = action_id
        self.action_type = action_type
        self.params = params
        self.selector = selector
        self.required_permissions = required_permissions or []
    
    def to_dict(self) -> dict:
        return {
            "id": self.action_id,
            "type": self.action_type.value,
            "params": self.params,
            "selector": self.selector,
            "required_permissions": self.required_permissions,
        }


class WorkflowStep:
    """Represents a step in a workflow"""
    
    def __init__(self,
                 step_id: str,
                 step_type: StepType,
                 content: Dict[str, Any],
                 condition: Optional[str] = None):
        self.step_id = step_id
        self.step_type = step_type
        self.content = content
        self.condition = condition
    
    def to_dict(self) -> dict:
        return {
            "id": self.step_id,
            "type": self.step_type.value,
            "content": self.content,
            "condition": self.condition,
        }


class WorkflowPlan:
    """Represents a complete workflow plan"""
    
    def __init__(self,
                 workflow_id: str,
                 goal: str,
                 steps: List[WorkflowStep],
                 required_permissions: List[str],
                 risk_assessment: Dict[str, Any],
                 estimated_duration_seconds: int = 120,
                 description: Optional[str] = None):
        self.workflow_id = workflow_id
        self.goal = goal
        self.steps = steps
        self.required_permissions = required_permissions
        self.risk_assessment = risk_assessment
        self.estimated_duration_seconds = estimated_duration_seconds
        self.description = description or self._generate_description()
        self.created_at = datetime.utcnow().isoformat()
    
    def _generate_description(self) -> str:
        """Generate a human-readable description of the workflow"""
        step_descriptions = []
        for step in self.steps:
            if step.step_type == StepType.ACTION:
                action_type = step.content.get("action_type", "unknown")
                step_descriptions.append(f"- {action_type}")
            elif step.step_type == StepType.DECISION:
                condition = step.condition or "check condition"
                step_descriptions.append(f"- If {condition}")
            elif step.step_type == StepType.CLARIFICATION:
                question = step.content.get("question", "ask user")
                step_descriptions.append(f"- Ask: {question}")
            elif step.step_type == StepType.SYNTHESIZE:
                step_descriptions.append("- Synthesize results")
        
        return "\n".join(step_descriptions)
    
    def to_dict(self) -> dict:
        return {
            "id": self.workflow_id,
            "goal": self.goal,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "required_permissions": self.required_permissions,
            "risk_assessment": self.risk_assessment,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "created_at": self.created_at,
        }


class PlanningEngine:
    """Plans workflows from user goals"""
    
    def __init__(self, model_generate_func=None):
        """
        Initialize planning engine
        
        Args:
            model_generate_func: Function to call for AI model inference
                                 Signature: generate_func(messages: list) -> str
        """
        self.model_generate_func = model_generate_func
        self.action_counter = 0
        self.workflow_counter = 0
        self.step_counter = 0
    
    def plan_workflow(self, goal: str, context: Dict[str, Any]) -> WorkflowPlan:
        """
        Generate a workflow plan from a user goal
        
        Args:
            goal: User's goal (e.g., "Compare RTX 5080 prices at 3 online stores")
            context: Browser context with current page, open tabs, etc.
        
        Returns:
            WorkflowPlan: Detailed workflow plan with steps and permissions
        """
        # Parse the goal
        parsed_goal = self._parse_goal(goal)
        
        # Generate action plan
        actions = self._generate_actions(parsed_goal, context)
        
        # Build workflow steps
        steps = self._build_workflow_steps(actions, parsed_goal)
        
        # Analyze required permissions
        required_permissions = self._analyze_permissions(steps)
        
        # Assess risks
        risk_assessment = self._assess_risk(steps, required_permissions)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(steps)
        
        # Create workflow plan
        self.workflow_counter += 1
        workflow_id = f"wf_{self.workflow_counter}_{int(datetime.utcnow().timestamp())}"
        
        workflow = WorkflowPlan(
            workflow_id=workflow_id,
            goal=goal,
            steps=steps,
            required_permissions=required_permissions,
            risk_assessment=risk_assessment,
            estimated_duration_seconds=estimated_duration,
            description=None,
        )
        
        return workflow
    
    def _parse_goal(self, goal: str) -> Dict[str, Any]:
        """Parse user goal into structured components"""
        goal = goal.strip()
        
        # Detect multi-site workflows
        is_comparison = any(word in goal.lower() for word in ["compare", "check", "find", "search"])
        is_multi_site = any(word in goal.lower() for word in ["at", "in", "from", "across"])
        
        # Extract key entities
        entities = self._extract_entities(goal)
        
        # Detect required actions
        required_actions = self._detect_required_actions(goal)
        
        return {
            "original_goal": goal,
            "is_comparison": is_comparison,
            "is_multi_site": is_multi_site,
            "entities": entities,
            "required_actions": required_actions,
            "complexity": "high" if is_multi_site else "medium" if is_comparison else "low",
        }
    
    def _extract_entities(self, goal: str) -> List[str]:
        """Extract key entities (products, sites, etc.) from goal"""
        entities = []
        
        # Look for quoted text
        quoted = re.findall(r'"([^"]+)"', goal)
        entities.extend(quoted)
        
        # Look for numbers
        numbers = re.findall(r'\b\d+\b', goal)
        entities.extend(numbers)
        
        # Look for URLs
        urls = re.findall(r'https?://[^\s]+', goal)
        entities.extend(urls)
        
        return entities
    
    def _detect_required_actions(self, goal: str) -> List[str]:
        """Detect which action types are needed"""
        actions = []
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ["read", "get", "extract", "find", "check"]):
            actions.append(ActionType.READ.value)
        if any(word in goal_lower for word in ["click", "select", "choose"]):
            actions.append(ActionType.CLICK.value)
        if any(word in goal_lower for word in ["enter", "type", "fill", "search"]):
            actions.append(ActionType.FILL.value)
        if any(word in goal_lower for word in ["go to", "visit", "navigate", "open"]):
            actions.append(ActionType.NAVIGATE.value)
        if any(word in goal_lower for word in ["scroll", "down", "up"]):
            actions.append(ActionType.SCROLL.value)
        if any(word in goal_lower for word in ["wait", "load", "appear"]):
            actions.append(ActionType.WAIT.value)
        if any(word in goal_lower for word in ["screenshot", "capture", "image"]):
            actions.append(ActionType.SCREENSHOT.value)
        
        return actions or [ActionType.READ.value, ActionType.NAVIGATE.value]
    
    def _generate_actions(self, parsed_goal: Dict[str, Any], context: Dict[str, Any]) -> List[ActionPlan]:
        """Generate action sequence from parsed goal"""
        actions = []
        required_actions = parsed_goal["required_actions"]
        
        # Build action sequence based on detected actions
        step_num = 1
        
        # If navigating, start with navigation
        if ActionType.NAVIGATE.value in required_actions:
            actions.append(ActionPlan(
                action_id=f"a_{step_num}",
                action_type=ActionType.NAVIGATE,
                params={"url": ""},  # Will be filled in by AI or user
                required_permissions=["navigate_pages"],
            ))
            step_num += 1
        
        # Search/fill actions
        if ActionType.FILL.value in required_actions:
            actions.append(ActionPlan(
                action_id=f"a_{step_num}",
                action_type=ActionType.FILL,
                params={"value": ""},  # To be determined
                selector="input[type='search'], input[type='text']",
                required_permissions=["fill_forms"],
            ))
            step_num += 1
        
        # Click actions
        if ActionType.CLICK.value in required_actions:
            actions.append(ActionPlan(
                action_id=f"a_{step_num}",
                action_type=ActionType.CLICK,
                params={},
                selector="button, a",
                required_permissions=["click_elements"],
            ))
            step_num += 1
        
        # Read/extract data
        actions.append(ActionPlan(
            action_id=f"a_{step_num}",
            action_type=ActionType.READ,
            params={"format": "text"},
            required_permissions=["current_page"],
        ))
        
        return actions
    
    def _build_workflow_steps(self, actions: List[ActionPlan], parsed_goal: Dict[str, Any]) -> List[WorkflowStep]:
        """Convert actions into workflow steps"""
        steps = []
        step_num = 1
        
        # For multi-site workflows, add loop structure
        if parsed_goal["is_multi_site"]:
            # Start with clarification if needed
            if not parsed_goal["entities"]:
                steps.append(WorkflowStep(
                    step_id=f"s_{step_num}",
                    step_type=StepType.CLARIFICATION,
                    content={
                        "question": "Which sites would you like me to check?",
                        "input_type": "text",
                    },
                ))
                step_num += 1
        
        # Convert actions to steps
        for action in actions:
            steps.append(WorkflowStep(
                step_id=f"s_{step_num}",
                step_type=StepType.ACTION,
                content={
                    "action_type": action.action_type.value,
                    "params": action.params,
                    "selector": action.selector,
                },
            ))
            step_num += 1
        
        # Add synthesis step
        steps.append(WorkflowStep(
            step_id=f"s_{step_num}",
            step_type=StepType.SYNTHESIZE,
            content={
                "description": "Compile and format results",
                "output_format": "comparison" if parsed_goal["is_comparison"] else "summary",
            },
        ))
        
        return steps
    
    def _analyze_permissions(self, steps: List[WorkflowStep]) -> List[str]:
        """Determine which permissions are required"""
        permissions = set()
        
        for step in steps:
            if step.step_type == StepType.ACTION:
                action_type = step.content.get("action_type")
                
                if action_type == ActionType.NAVIGATE.value:
                    permissions.add("navigate_pages")
                elif action_type == ActionType.CLICK.value:
                    permissions.add("click_elements")
                elif action_type == ActionType.FILL.value:
                    permissions.add("fill_forms")
                elif action_type == ActionType.READ.value:
                    permissions.add("current_page")
                    permissions.add("selected_text")
                elif action_type == ActionType.SCROLL.value:
                    permissions.add("current_page")
                elif action_type == ActionType.SCREENSHOT.value:
                    permissions.add("current_page")
        
        return sorted(list(permissions))
    
    def _assess_risk(self, steps: List[WorkflowStep], permissions: List[str]) -> Dict[str, Any]:
        """Assess risk level of workflow"""
        critical_actions = 0
        high_risk_actions = 0
        medium_risk_actions = 0
        
        for step in steps:
            if step.step_type == StepType.ACTION:
                action_type = step.content.get("action_type")
                
                if action_type in [ActionType.CLICK.value, ActionType.FILL.value]:
                    high_risk_actions += 1
                elif action_type == ActionType.NAVIGATE.value:
                    medium_risk_actions += 1
        
        # Determine overall risk
        if critical_actions > 0:
            overall_risk = RiskLevel.CRITICAL
        elif high_risk_actions > 2:
            overall_risk = RiskLevel.HIGH
        elif high_risk_actions > 0 or medium_risk_actions > 3:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW
        
        return {
            "overall": overall_risk.value,
            "critical_actions": critical_actions,
            "high_risk_actions": high_risk_actions,
            "medium_risk_actions": medium_risk_actions,
            "requires_approval": overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            "description": self._risk_description(overall_risk),
        }
    
    def _risk_description(self, risk: RiskLevel) -> str:
        """Generate risk description"""
        descriptions = {
            RiskLevel.LOW: "This workflow is low-risk and uses read-only operations.",
            RiskLevel.MEDIUM: "This workflow is moderate risk with some interactive actions.",
            RiskLevel.HIGH: "This workflow is high-risk and performs multiple interactive operations.",
            RiskLevel.CRITICAL: "This workflow is critical risk and requires careful review.",
        }
        return descriptions.get(risk, "Unknown risk level")
    
    def _estimate_duration(self, steps: List[WorkflowStep]) -> int:
        """Estimate workflow duration in seconds"""
        base_time = 5
        per_action_time = 3
        wait_time = 2
        
        action_count = sum(1 for step in steps if step.step_type == StepType.ACTION)
        wait_count = sum(1 for step in steps if step.content.get("action_type") == ActionType.WAIT.value)
        
        estimated = base_time + (action_count * per_action_time) + (wait_count * wait_time)
        
        # Add buffer for synthesis
        if any(step.step_type == StepType.SYNTHESIZE for step in steps):
            estimated += 5
        
        return max(10, min(300, estimated))  # Between 10s and 5 minutes


class ObservationAnalyzer:
    """Analyzes observations from action execution"""
    
    def analyze_observation(self, observation: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an observation and determine if plan adaptation is needed
        
        Args:
            observation: Observation from action execution
            step: Current workflow step
        
        Returns:
            Analysis including success status and adaptation suggestions
        """
        analysis = {
            "success": observation.get("success", False),
            "needs_adaptation": False,
            "suggested_actions": [],
            "error_message": None,
        }
        
        if not analysis["success"]:
            analysis["error_message"] = observation.get("error")
            
            # Determine adaptation strategy
            if "not found" in str(observation.get("error", "")).lower():
                analysis["suggested_actions"] = ["retry_with_alt_selector", "wait_and_retry"]
                analysis["needs_adaptation"] = True
            elif "timeout" in str(observation.get("error", "")).lower():
                analysis["suggested_actions"] = ["wait_longer", "reload_page"]
                analysis["needs_adaptation"] = True
            elif "permission" in str(observation.get("error", "")).lower():
                analysis["suggested_actions"] = ["request_permission", "clarify_user"]
                analysis["needs_adaptation"] = True
        
        return analysis


class ResultSynthesizer:
    """Synthesizes results from workflow execution"""
    
    def synthesize(self, observations: List[Dict[str, Any]], goal: str) -> Dict[str, Any]:
        """
        Synthesize workflow observations into results
        
        Args:
            observations: List of observations from workflow steps
            goal: Original user goal
        
        Returns:
            Synthesized results
        """
        result = {
            "goal": goal,
            "success": all(obs.get("success", False) for obs in observations),
            "execution_time": sum(obs.get("timing", {}).get("total_ms", 0) for obs in observations),
            "observations_count": len(observations),
            "extracted_data": self._extract_data(observations),
            "summary": self._generate_summary(observations, goal),
        }
        
        return result
    
    def _extract_data(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant data from observations"""
        extracted = {
            "text_content": [],
            "links": [],
            "forms": [],
            "images": [],
        }
        
        for obs in observations:
            result = obs.get("result", {})
            
            if isinstance(result, dict):
                if "text" in result:
                    extracted["text_content"].append(result["text"])
                if "elements_found" in result:
                    extracted["forms"].append(result)
        
        return extracted
    
    def _generate_summary(self, observations: List[Dict[str, Any]], goal: str) -> str:
        """Generate summary of workflow execution"""
        successful_steps = sum(1 for obs in observations if obs.get("success", False))
        total_steps = len(observations)
        
        return f"Completed {successful_steps}/{total_steps} steps. " \
               f"Goal: {goal}. " \
               f"Total execution time: {sum(obs.get('timing', {}).get('total_ms', 0) for obs in observations)}ms"
