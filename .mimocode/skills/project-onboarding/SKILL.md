---
name: project-onboarding
description: "Structured workflow for understanding and continuing existing projects"
---

# Project Onboarding Skill

This skill provides a systematic approach for understanding and continuing existing projects, with emphasis on efficient context absorption and immediate productivity.

## Usage

```
/project-onboarding <project_directory> [options]
```

### Parameters

- `project_directory`: Path to the project root (required)
- `--context-file <path>`: Path to context/handoff document (optional)
- `--skip-audit`: Skip completed phase audits (optional)
- `--language <lang>`: Communication language (default: Chinese)

## Workflow

### Phase 1: Context Absorption

1. **Read documentation first**
   - Project overview/README
   - Handoff documents
   - Implementation plans
   - Status reports

2. **Understand project structure**
   - Directory layout
   - Key modules
   - Configuration files
   - Test structure

3. **Review recent changes**
   - Git history
   - Recent commits
   - Active branches
   - Pending changes

### Phase 2: Code Understanding

1. **Core module analysis**
   - Main entry points
   - Key classes/functions
   - Data flow
   - API interfaces

2. **Test coverage review**
   - Existing tests
   - Test patterns
   - Coverage gaps
   - Test utilities

3. **Configuration analysis**
   - Environment setup
   - Dependencies
   - Build system
   - Deployment config

### Phase 3: Status Assessment

1. **Identify completed work**
   - Finished features
   - Resolved issues
   - Completed tests
   - Documentation updates

2. **Identify pending work**
   - TODO items
   - Incomplete features
   - Known issues
   - Technical debt

3. **Prioritize tasks**
   - Critical path items
   - Quick wins
   - Dependencies
   - Risk assessment

### Phase 4: Immediate Productivity

1. **Start with quick wins**
   - Simple fixes
   - Documentation updates
   - Test additions
   - Minor improvements

2. **Build momentum**
   - Complete small tasks first
   - Verify each change
   - Update documentation
   - Commit frequently

3. **Maintain continuity**
   - Follow existing patterns
   - Respect conventions
   - Preserve style
   - Keep consistency

## Communication Protocol

### Status Reports
- **What changed**: Brief description of modifications
- **How verified**: Testing/verification steps
- **Next steps**: Immediate follow-up actions

### Progress Updates
- Use Chinese for communication (unless specified otherwise)
- Be concise and actionable
- Focus on results, not process
- Highlight blockers immediately

## Best Practices

1. **Read before writing** - Understand existing code before modifying
2. **Verify before committing** - Test changes before saving
3. **Document as you go** - Update docs with changes
4. **Communicate progress** - Regular status updates
5. **Ask when blocked** - Don't waste time on unclear requirements

## Context Files

The skill looks for these context files in the project directory:

- `HANDOFF_OVERVIEW.md` - Project handoff summary
- `implementation-plans/` - Implementation plans directory
- `TODO.md` - Task list
- `CHANGELOG.md` - Change history
- `docs/` - Documentation directory

## Output

The skill should provide:
- Project understanding summary
- Current status assessment
- Prioritized task list
- Immediate action plan
- Progress updates