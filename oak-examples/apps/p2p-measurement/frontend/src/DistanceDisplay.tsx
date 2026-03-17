import { css } from "../styled-system/css/css.mjs";
import { useState } from "react";

interface DistanceDisplayProps {
    distance: number | null;
    stdDeviation?: number | null;
    pointCount?: number;
    hasInvalidDepth?: boolean;
    trackingEnabled?: boolean;
    onToggleTracking?: () => void;
}

export function DistanceDisplay({ distance, stdDeviation, pointCount = 0, hasInvalidDepth = false, trackingEnabled = true, onToggleTracking }: DistanceDisplayProps) {
    const [unitSystem, setUnitSystem] = useState<'metric' | 'imperial'>('metric');
    const [rounding, setRounding] = useState<1 | 2 | 3 | 4>(3);
    
    const formatDistance = (dist: number) => {
        if (unitSystem === 'metric') {
            if (dist < 0.01) {
                return `${(dist * 1000).toFixed(rounding === 1 ? 0 : rounding - 1)} mm`;
            } else if (dist < 1) {
                return `${(dist * 100).toFixed(rounding === 1 ? 0 : rounding - 1)} cm`;
            } else {
                return `${dist.toFixed(rounding)} m`;
            }
        } else {
            // Imperial system
            const feet = dist * 3.28084;
            if (feet < 1) {
                const inches = feet * 12;
                return `${inches.toFixed(rounding === 1 ? 0 : rounding - 1)} in`;
            } else if (feet < 5280) {
                return `${feet.toFixed(rounding)} ft`;
            } else {
                const miles = feet / 5280;
                return `${miles.toFixed(rounding)} mi`;
            }
        }
    };

    return (
        <div className={css({
            padding: 'sm',
            backgroundColor: 'white',
            borderRadius: 'md',
            border: distance !== null ? '2px solid' : 'none',
            borderColor: distance !== null ? 'black' : 'transparent',
            boxShadow: distance !== null ? '0 4px 12px rgba(0, 0, 0, 0.15)' : '0 2px 8px rgba(0, 0, 0, 0.08)',
            textAlign: 'center'
        })}>            
            {/* Controls */}
            <div className={css({
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: 'md',
                marginBottom: 'lg'
            })}>
                {/* Unit System Toggle */}
                <div className={css({ display: 'flex', gap: 'xs' })}>
                    <button
                        className={css({
                            padding: 'xs',
                            fontSize: 'sm',
                            backgroundColor: 'transparent',
                            color: unitSystem === 'metric' ? 'black' : 'gray.400',
                            textDecoration: unitSystem === 'metric' ? 'underline' : 'none',
                            border: 'none',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                        })}
                        onClick={() => setUnitSystem('metric')}
                    >
                        metric
                    </button>
                    <button
                        className={css({
                            padding: 'xs',
                            fontSize: 'sm',
                            backgroundColor: 'transparent',
                            color: unitSystem === 'imperial' ? 'black' : 'gray.400',
                            textDecoration: unitSystem === 'imperial' ? 'underline' : 'none',
                            border: 'none',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                        })}
                        onClick={() => setUnitSystem('imperial')}
                    >
                        imperial
                    </button>
                </div>
                
                {/* Tracking Toggle */}
                {onToggleTracking && (
                    <button
                        className={css({
                            padding: 'xs',
                            fontSize: 'xs',
                            border: '1px solid black',
                            backgroundColor: trackingEnabled ? 'black' : 'transparent',
                            color: trackingEnabled ? 'white' : 'black',
                            borderRadius: 'sm',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                        })}
                        onClick={onToggleTracking}
                    >
                        {trackingEnabled ? 'tracking' : 'not tracking'}
                    </button>
                )}
                
                {/* Rounding Options */}
                <div className={css({ display: 'flex', gap: 'xs' })}>
                    {[1, 2, 3, 4].map((decimals) => (
                        <button
                            key={decimals}
                            className={css({
                                padding: 'xs',
                                fontSize: 'xs',
                                border: '1px solid black',
                                backgroundColor: rounding === decimals ? 'black' : 'transparent',
                                color: rounding === decimals ? 'white' : 'black',
                                borderRadius: 'sm',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                                width: '24px',
                                height: '24px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            })}
                            onClick={() => setRounding(decimals as 1 | 2 | 3 | 4)}
                        >
                            {decimals}
                        </button>
                    ))}
                </div>
            </div>

            {distance !== null ? (
                <div>
                    <div className={css({ 
                        fontSize: '2xl', 
                        fontWeight: 'bold',
                        color: 'green.700',
                        marginBottom: 'lg'
                    })}>
                        {formatDistance(distance)}
                        {stdDeviation !== null && stdDeviation !== undefined && stdDeviation > 0 && (
                            <span className={css({ 
                                fontSize: 'lg',
                                fontWeight: 'normal',
                                color: 'green.600',
                                marginLeft: 'xs'
                            })}>
                                ±{formatDistance(stdDeviation)}
                            </span>
                        )}
                    </div>
                    {hasInvalidDepth && (
                        <div className={css({ 
                            fontSize: 'sm',
                            color: 'red.600',
                            backgroundColor: 'red.50',
                            padding: 'xs',
                            borderRadius: 'sm',
                            border: '1px solid',
                            borderColor: 'red.200',
                            marginBottom: 'xs',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 'xs'
                        })}>
                            <span>⚠️</span>
                            <span>Invalid depth detected - try selecting different points</span>
                        </div>
                    )}
                    <div className={css({ 
                        fontSize: 'sm',
                        color: 'gray.600'
                    })}>
                        3D Euclidean distance
                    </div>
                </div>
            ) : pointCount === 2 ? (
                <div>
                    <div className={css({ 
                        fontSize: 'lg',
                        color: 'orange.700',
                        marginBottom: 'xs'
                    })}>
                        ⏳ Calculating distance...
                    </div>
                    {hasInvalidDepth && (
                        <div className={css({ 
                            fontSize: 'sm',
                            color: 'red.600',
                            backgroundColor: 'red.50',
                            padding: 'xs',
                            borderRadius: 'sm',
                            border: '1px solid',
                            borderColor: 'red.200',
                            marginBottom: 'xs',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 'xs'
                        })}>
                            <span>⚠️</span>
                            <span>Invalid depth detected - try selecting different points</span>
                        </div>
                    )}
                    <div className={css({ 
                        fontSize: 'sm',
                        color: 'gray.600'
                    })}>
                        Two points selected
                    </div>
                </div>
            ) : (
                <div className={css({ 
                    fontSize: 'md',
                    color: 'gray.500',
                    marginBottom: 'lg'
                })}>
                    Select two points to measure
                </div>
            )}
        </div>
    );
}
