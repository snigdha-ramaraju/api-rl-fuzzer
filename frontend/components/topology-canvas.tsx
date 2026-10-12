'use client'
import { useCallback } from 'react'
import { Background, Controls, MiniMap, ReactFlow, addEdge, useEdgesState, useNodesState, type Connection, type Edge, type Node } from '@xyflow/react'
import '@xyflow/react/dist/style.css'

const initialNodes: Node[] = [{ id: '1', position: { x: 40, y: 90 }, data: { label: 'POST /orders' }, type: 'input' }, { id: '2', position: { x: 280, y: 40 }, data: { label: 'GET /orders/{id}' } }, { id: '3', position: { x: 280, y: 160 }, data: { label: 'GET /customers/{id}' } }, { id: '4', position: { x: 540, y: 90 }, data: { label: 'POST /payments' }, className: 'finding-node' }, { id: '5', position: { x: 760, y: 200 }, data: { label: 'GET /payments/{id}' } }]
const initialEdges: Edge[] = [{ id: 'e1-2', source: '1', target: '2', animated: true }, { id: 'e1-3', source: '1', target: '3' }, { id: 'e2-4', source: '2', target: '4' }, { id: 'e4-5', source: '4', target: '5' }]

export function TopologyCanvas() { const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes); const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges); const onConnect = useCallback((connection: Connection) => setEdges((current) => addEdge(connection, current)), [setEdges]); return <div className="flow-wrap"><ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} fitView><MiniMap /><Controls /><Background color="#dce6e1" gap={24} /></ReactFlow><div className="flow-note">Drag nodes · Connect routes to extend the demo map</div></div> }
