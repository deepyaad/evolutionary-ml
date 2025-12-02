#!/usr/bin/env python3
"""
Script to check template status and update it if needed.
"""
import os
import runpod
import sys

def main():
    # Get API key
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        api_key_paths = ['../../api-key.txt', 'api-key.txt']
        for path in api_key_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    api_key = f.read().strip()
                break
    
    if not api_key:
        print("ERROR: Need RUNPOD_API_KEY")
        sys.exit(1)
    
    runpod.api_key = api_key
    
    # Get template ID (from previous deployment or environment)
    template_id = os.getenv('RUNPOD_TEMPLATE_ID', '0d4lvflx5m')  # Your template ID
    
    print(f"Checking template: {template_id}")
    print("="*60)
    
    # Get all templates to see if ours exists
    try:
        # Note: get_templates might not exist, so we'll try a different approach
        # Get endpoints and check their templates
        endpoints = runpod.get_endpoints()
        
        print("\nYour Endpoints:")
        print("-" * 60)
        for endpoint in endpoints:
            if isinstance(endpoint, dict):
                ep_id = endpoint.get('id', 'unknown')
                ep_name = endpoint.get('name', 'unknown')
                ep_template = endpoint.get('templateId', endpoint.get('template_id', 'unknown'))
                print(f"  Endpoint: {ep_name} (ID: {ep_id})")
                print(f"    Template ID: {ep_template}")
                if ep_template == template_id:
                    print(f"    ✅ This endpoint uses your template!")
            else:
                print(f"  Endpoint: {endpoint}")
        
        print("\n" + "="*60)
        print("To update your template:")
        print("1. Rebuild your Docker image (if code changed)")
        print("2. Create a new template with the new image")
        print("3. Update your endpoint to use the new template")
        print("\nOr use the web dashboard:")
        print("  https://www.runpod.io/console/serverless")
        print("  → Go to your endpoint")
        print("  → Click 'Edit' or 'Update Template'")
        print("  → Select a new template or update the existing one")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nAlternative: Check RunPod web dashboard:")
        print("  https://www.runpod.io/console/serverless")
        print("  → Templates section to see all templates")
        print("  → Your endpoint to see which template it uses")

if __name__ == "__main__":
    main()

