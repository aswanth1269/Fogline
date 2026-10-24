---
version: alpha
name: Fogline
description: A local research console for adaptive edge-fog intrusion detection.
colors:
  primary: '#176b64'
  background: '#f3f6f8'
  surface: '#ffffff'
  text: '#1c3040'
  muted: '#617482'
  border: '#dce5ea'
  danger: '#b33c4b'
  success: '#176b64'
  warning: '#946316'
typography:
  body:
    fontFamily: 'Segoe UI, sans-serif'
  display:
    fontFamily: 'Bahnschrift, Segoe UI, sans-serif'
  mono:
    fontFamily: 'Cascadia Code, Consolas, monospace'
rounded:
  DEFAULT: '12px'
spacing:
  panel: '25px'
components:
  button: {}
  panel: {}
  input: {}
---
# Fogline design system
## Overview
A network operations workbench for students demonstrating adaptive IDS decisions. Product register, English interface, local laptop use. The signature is the horizontal traffic-to-edge-to-fog pathway; surrounding controls stay quiet. Avoid hacker-terminal theatrics, neon dashboards, and decorative architecture posters.
Runtime CSS is canonical (Model B): frontend/style.css :root tokens map directly to same-named colors above. Typography maps body/display/mono to --font-body/--font-display/--font-mono. Radius maps to --radius. Components consume those variables; there is no framework adapter or duplicate theme.
## Colors
Cool blue-gray workspace, white panels, deep marine navigation, teal primary actions. Risk uses both words and semantic colors. One light theme; forced-colors preserves system scrollbars and borders.
## Typography
Bahnschrift headings evoke instrument labels. Segoe UI handles forms and prose; Cascadia Code handles event identifiers. Local fonts avoid external requests and font-swap layout shifts.
## Layout
236px navigation at desktop; 195px below 1100px; horizontal navigation below 700px. Main content uses natural document scrolling. Tables own horizontal overflow. Panels use 25px padding and 23px gaps.
## Elevation & Depth
Borders and surface contrast define hierarchy. No static card shadows. Modal backdrop separates the detail layer.
## Shapes
12px panels, 7px controls, compact labeled risk badges.
## Components
Shared button styles cover hover, focus, pressed, disabled and busy semantics without label resizing. Shared api/busy/toast/table/result functions in frontend/app.js own repeated behavior. Native select geometry is accepted. Native dialog owns focus trapping and Escape. One live toast announces saves; inline errors preserve recovery details. Meters always include textual values. Motion is limited to button press; reduced motion is respected.
## Do's and Don'ts
- Show real measured values and explicit empty states.
- Keep simulated inputs and recommended responses honestly labeled.
- Do not present heuristic feature indicators as SHAP.
- Do not imply real network blocking or distributed federated learning.
