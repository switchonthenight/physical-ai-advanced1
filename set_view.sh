#!/bin/bash
# 使い方: ./set_view.sh A|B|C
case "$1" in
  A) P="x: -0.14, y: 0.0, z: 6.6";  Q="w: 0.88295, x: 0.0, y: 0.46947, z: 0.0" ;;
  B) P="x: 7.0, y: 0.0, z: 12.0";   Q="w: 0.5, x: -0.5, y: 0.5, z: 0.5" ;;
  C) P="x: 7.0, y: -5.9, z: 7.0";   Q="w: 0.6409, x: -0.2988, y: 0.2988, z: 0.6409" ;;
  *) echo "usage: $0 A|B|C"; exit 1 ;;
esac
ign service -s /gui/move_to/pose \
  --reqtype ignition.msgs.GUICamera --reptype ignition.msgs.Boolean --timeout 2000 \
  --req "pose: {position: {$P}, orientation: {$Q}}"