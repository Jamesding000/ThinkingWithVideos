#!/bin/bash
# Quick Memory Leak Diagnostic Script
# Run this while your training is running

echo "========================================="
echo "Quick Memory Leak Diagnostic"
echo "========================================="
echo ""

echo "1. Checking Ray processes..."
ps aux | grep ray | grep -v grep | awk '{print $2, $4, $11}' | head -20

echo ""
echo "2. Ray Object Store Status:"
python check_ray_memory_alternative.py 2>/dev/null || echo "  (Could not check Ray memory - see check_ray_memory_alternative.py)"

echo ""
echo "3. Top memory consuming Python processes:"
ps aux | grep python | grep -v grep | sort -k4 -rn | head -10 | awk '{printf "  PID: %s  MEM: %s%%  CMD: %s\n", $2, $4, $11}'

echo ""
echo "4. System memory status:"
free -h

echo ""
echo "5. Watching for growth (monitoring for 60 seconds)..."
echo "   Taking snapshot..."

# Take initial snapshot
INITIAL=$(ps aux | grep "TaskRunner\|WorkerDict" | grep -v grep | awk '{sum+=$4} END {print sum}')
echo "   Initial memory: ${INITIAL}%"

sleep 60

# Take final snapshot
FINAL=$(ps aux | grep "TaskRunner\|WorkerDict" | grep -v grep | awk '{sum+=$4} END {print sum}')
echo "   Final memory: ${FINAL}%"

GROWTH=$(echo "$FINAL - $INITIAL" | bc 2>/dev/null || echo "N/A")
echo "   Growth: ${GROWTH}%"

echo ""
echo "========================================="
echo "Recommendations:"
echo "========================================="

if (( $(echo "$GROWTH > 1" | bc -l 2>/dev/null) )); then
    echo "  ⚠️  Significant memory growth detected!"
    echo "  Next steps:"
    echo "    1. Run: python debug_memory_leak.py (for detailed analysis)"
    echo "    2. Check: ray memory (for object store)"
    echo "    3. Review: DEEP_MEMORY_DEBUG_GUIDE.md"
else
    echo "  ✅ No significant growth in last 60s"
    echo "  If you're still seeing leaks over hours:"
    echo "    1. Run: python debug_memory_leak.py (for longer monitoring)"
    echo "    2. Check for slow accumulation patterns"
fi

echo ""

