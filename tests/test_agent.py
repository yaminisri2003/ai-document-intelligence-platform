# test_agent.py
# Run with: python -m tests.test_agent
#
# This test demonstrates two things:
# 1. The LangGraph routing — simple vs document questions
# 2. Multi-turn memory — follow-up questions work correctly

from app.agents.graph import ConversationAgent


def test_agent():

    print("\n=== Testing LangGraph Conversation Agent ===\n")

    # Create agent for HR domain
    agent = ConversationAgent(domain="hr_enterprise")

    # Test 1: Simple greeting — should skip retrieval
    print("--- Test 1: Simple Greeting (no retrieval needed) ---")
    result = agent.chat("Hello!")
    print(f"Question: Hello!")
    print(f"Answer: {result['answer']}")
    print(f"Sources used: {len(result['sources'])}")
    print()

    # Test 2: Document question — should retrieve
    print("--- Test 2: Document Question (retrieval needed) ---")
    result = agent.chat("What are the core working hours?")
    print(f"Question: What are the core working hours?")
    print(f"Answer: {result['answer']}")
    print(f"Sources used: {len(result['sources'])}")
    print()

    # Test 3: Follow-up question — tests memory
    # This question refers to the previous answer
    # Without memory the agent cannot answer this correctly
    print("--- Test 3: Follow-up Question (tests memory) ---")
    result = agent.chat("What if I am in India timezone?")
    print(f"Question: What if I am in India timezone?")
    print(f"Answer: {result['answer']}")
    print()

    # Test 4: Another document question
    print("--- Test 4: Internet reimbursement ---")
    result = agent.chat("Will the company pay for my internet?")
    print(f"Question: Will the company pay for my internet?")
    print(f"Answer: {result['answer']}")
    print()

    # Show conversation history length
    print(f"Total conversation turns stored: {len(agent.chat_history)}")
    print("\n=== Agent Test Complete ===")


if __name__ == "__main__":
    test_agent()