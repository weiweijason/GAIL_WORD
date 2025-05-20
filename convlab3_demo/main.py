import convlab
# NLU component (from convlab/nlu/svm/)
from convlab.nlu.svm.multiwoz import SVMNLU
# Policy component (from convlab/policy/mle/)
# from convlab.policy.mle.multiwoz import MLEPolicy # Previous attempt
from convlab.policy.mle import MLEPolicy # Updated import
# NLG component (from convlab/nlg/template/)
from convlab.nlg.template.multiwoz import TemplateNLG
# Optional: For more detailed logging from ConvLab components
# import logging
# logging.basicConfig(level=logging.INFO) # Or logging.DEBUG for very verbose output

# Existing example comments for CamRest are kept below for reference,
# but the new demo will use MultiWOZ components.
# Example: Load a pre-trained model (replace with a specific model)
# try:
#     nlu = convlab.nlu.svm.camrest.nlu.SVMNLU()
#     print("NLU model loaded successfully.")
# except Exception as e:
#     print(f"Error loading NLU model: {e}")
#     print("Please ensure you have downloaded the necessary model files or try a different one.")
# ... (other CamRest examples remain commented out) ...

def run_simulated_dialogue():
    print("Initializing ConvLab-3 components for MultiWOZ...")
    try:
        # 1. Natural Language Understanding (NLU)
        print("Loading NLU model (SVMNLU for MultiWOZ)...")
        # Located in: convlab/nlu/svm/
        # Example download commands (run in your terminal):
        # python -m convlab.util.download multiwoz
        # python -m convlab.util.download multiwoz_svm_nlu
        nlu = SVMNLU()
        print("NLU model loaded.")

        # 2. Dialogue Policy
        print("Loading Policy model (MLEPolicy for MultiWOZ)...") # Using MLEPolicy
        # Located in: convlab/policy/mle/
        # Example download command for MLEPolicy (check ConvLab-3 docs for exact command):
        # python -m convlab.util.download multiwoz_mle_policy
        policy = MLEPolicy() # Using MLEPolicy
        print("Policy model loaded.")

        # 3. Natural Language Generation (NLG)
        print("Loading NLG model (TemplateNLG for MultiWOZ)...")
        # Located in: convlab/nlg/template/
        nlg = TemplateNLG(is_user=False) # System NLG
        print("NLG model loaded.")

    except Exception as e:
        print(f"Error initializing ConvLab-3 components: {e}")
        print("Please ensure you have installed ConvLab-3 correctly and downloaded any necessary models.")
        print("For MultiWOZ models, you might need to run commands like (in your terminal):")
        print("  `python -m convlab.util.download multiwoz` (for the dataset)")
        print("  `python -m convlab.util.download multiwoz_svm_nlu` (for SVMNLU)")
        print("  `python -m convlab.util.download multiwoz_mle_policy` (for MLEPolicy - check exact name)")
        print("Refer to ConvLab-3 documentation for specific model requirements and download commands.")
        return

    print("\n--- Starting Simulated Dialogue Session (MultiWOZ) ---")
    print("Type 'bye' to end the session.")

    policy.init_session() # Initialize the policy's internal state for a new session

    while True:
        user_utterance = input("You: ")
        if user_utterance.lower() == 'bye':
            print("System: Goodbye!")
            break

        try:
            # 1. NLU: Process user utterance
            # SVMNLU().predict typically takes the utterance. Context can be added if needed.
            user_action = nlu.predict(user_utterance)
            print(f"DEBUG: NLU recognized user action: {user_action}")
        except Exception as e:
            print(f"NLU Error: {e}")
            print("System: I had trouble understanding that.")
            continue

        try:
            # 2. Policy: Get system action
            # RulePolicy's predict method takes the user_action (as observation)
            # and updates its internal state.
            system_action = policy.predict(user_action)
            print(f"DEBUG: Policy decided system action: {system_action}")
        except Exception as e:
            print(f"Policy Error: {e}")
            print("System: I'm not sure how to respond to that.")
            continue

        try:
            # 3. NLG: Generate system response
            system_response = nlg.generate(system_action)
            print(f"System: {system_response}")
        except Exception as e:
            print(f"NLG Error: {e}")
            # Fallback response if NLG fails
            if system_action:
                simple_response_parts = []
                for act_item in system_action: # system_action is a list of dialogue acts
                    # Ensure unpacking handles various act formats, taking first 4 elements
                    act_elements = list(act_item) + [None] * (4 - len(act_item))
                    act_type, domain, slot, value = act_elements[:4]

                    if value not in ['?', 'none', 'dontcare', None] and value : # Check for meaningful value
                         simple_response_parts.append(f"{domain} {slot} is {value}")
                    elif act_type and act_type.lower() == 'request' and domain and slot:
                        simple_response_parts.append(f"Could you tell me about {domain} {slot}?")
                    elif act_type and domain and slot: # Generic fallback for other actions
                        display_value = f" {value}" if value not in [None, '?', 'none', 'dontcare'] else ""
                        simple_response_parts.append(f"{act_type} {domain} {slot}{display_value}".strip())
                    elif act_type and domain : # Broader fallback
                         simple_response_parts.append(f"{act_type} for {domain}")
                if simple_response_parts:
                    print(f"System (fallback): {'. '.join(simple_response_parts)}")
                else:
                    print("System: I have a response, but couldn't phrase it clearly.")
            else:
                 print("System: I have an action, but couldn't generate a response text.")
            continue

if __name__ == "__main__":
    print("Welcome to the ConvLab-3 Demo Project!")
    print("This script demonstrates a simulated dialogue session with MultiWOZ components.")
    print("Ensure you have ConvLab-3 installed (pip install convlab3).")
    print("And necessary models downloaded (e.g., using 'python -m convlab.util.download multiwoz').")
    print("This example now uses SVMNLU, MLEPolicy, and TemplateNLG.")
    print("You might need to download 'multiwoz_svm_nlu' and 'multiwoz_mle_policy' (check exact names).")
    run_simulated_dialogue()
