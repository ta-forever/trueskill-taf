#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <vector>

namespace factorgraph
{
    class Message;

    class Factor
    {
    public:

        // initialise all input messages to "identity"
        virtual void Initialise() =0;

        // update any or all output messages
        virtual void Update(int which=0, const char *debug=0) =0;

        virtual const char* Type() const =0;

    protected:

        struct MessagePair
        {
            MessagePair(const Message *_in, Message *_out):
                mutable_in(0), const_in(_in), out(_out)
                { }

            MessagePair(Message *_in, Message *_out):
                mutable_in(_in), const_in(0), out(_out)
                { }

            const Message *In() const { return const_in ? const_in : mutable_in; }
            Message *MutableIn() { return mutable_in; }
            Message *Out() { return out; }

        private:
            const Message *const_in;
            Message *mutable_in;
            Message *out;
        };
    };


    // common framework for SumFactor and EqualityFactor
    class ThreeMessageFactor: public Factor
    {
    public:
        ThreeMessageFactor(
            Message *xin, Message *xout, 
            Message *yin, Message *yout, 
            Message *zin, Message *zout);

        ThreeMessageFactor(
            const Message *xin, Message *xout, 
            Message *yin, Message *yout, 
            Message *zin, Message *zout);

        virtual void Initialise();
        virtual void Update(int which=0, const char *debug=0);

    private:
        // m2 == f(m,m2)
        virtual void _Initialise(Message *m) const =0;

        // m1 = f(m2,k*m3)
        virtual void _Apply(Message *m1, const Message *m2, double k, const Message *m3, const char *debug) const =0;

        MessagePair m_x;
        MessagePair m_y;
        MessagePair m_z;
    };


    // z = x + y
    class SumFactor: public ThreeMessageFactor
    {
    public:
        SumFactor(
            Message *xin, Message *xout, 
            Message *yin, Message *yout, 
            Message *zin, Message *zout,
            bool initialise=true);

        SumFactor(
            const Message *xin, Message *xout, 
            Message *yin, Message *yout, 
            Message *zin, Message *zout,
            bool initialise=true);

        virtual const char* Type() const { return "SUM"; }

    private:
        virtual void _Initialise(Message *m) const;
        virtual void _Apply(Message *m1, const Message *m2, double k, const Message *m3, const char *debug) const;
    };

    // z = x - y
    class DifferenceFactor: public SumFactor
    {
    public:
        DifferenceFactor(
            Message *xin, Message *xout, 
            Message *yin, Message *yout, 
            Message *zin, Message *zout,
            bool initialise=true):
          SumFactor(zin,zout,yin,yout,xin,xout,initialise)
          { }

        virtual const char* Type() const { return "DIFFERENCE"; }

    };

    // y = sum(xi)
    class SumNFactor: public Factor
    {
    public:
        SumNFactor(Message *yin, Message *yout, bool initialise=true);
        void PushMessagePair(Message *xin, Message *xout, bool initialise=true);
        void PushMessagePair(const Message *xin, Message *xout);

        virtual void Initialise();
        virtual void Update(int which=0, const char *debug=0);
        virtual const char* Type() const { return "SUMN"; }

    private:
        std::vector<MessagePair> m_x;
        MessagePair m_y;
    };


    // x == y == z
    // ie a message splitter
    class EqualityFactor: public ThreeMessageFactor
    {
    public:
        EqualityFactor(
            Message *xin, Message *xout, 
            Message *yin, Message *yout, 
            Message *zin, Message *zout,
            bool initialise=true);

        virtual const char* Type() const { return "EQUAL"; }

    private:
        virtual void _Initialise(Message *m) const;
        virtual void _Apply(Message *m1, const Message *m2, double k, const Message *m3, const char *debug) const;
    };

    // lb <= x < ub
    // If you attach this to a SumFactor, make sure to either
    // - use finite lb and ub; or
    // - make sure input message is initialised with some finite amount of information prior to calling Update
    class TerminalConstraintFactor: public Factor
    {
    public:
        TerminalConstraintFactor(double lb, double ub, Message *min, Message *mout, bool initialise=true);

        virtual void Initialise();
        virtual void Update(int which=0, const char *debug=0);
        virtual const char* Type() const { return "TERMINAL_CONSTRAINT"; }

    private:
        double m_lb;
        double m_ub;

        MessagePair m_message;
    };

    // combination equality splitter and terminal constraint, for convenience
    // passes message X to message Y (and vice versa) with a constraint applied
    class InteriorConstraintFactor: public Factor
    {
    public:
        InteriorConstraintFactor(double lb, double ub, 
            Message *xin, Message *xout,
            Message *yin, Message *yout,
            bool constrainXin2Yout=true,
            bool constrainYin2Xout=true,
            bool initialise=true);

        virtual void Initialise();
        virtual void Update(int which=0, const char *debug=0);
        virtual const char* Type() const { return "INTERIOR_CONSTRAINT"; }

    private:
        void Update0(const MessagePair &u, MessagePair &v, bool doConstrain, bool perVariable, const char *debug);

        double m_lb;
        double m_ub;

        MessagePair m_x;
        MessagePair m_y;
        bool m_constrainXin2Yout;
        bool m_constrainYin2Xout;
    };

    class TransformFactor: public Factor
    {
    public:
        TransformFactor(const Eigen::MatrixXd &A,
            Message *xin, Message *xout,
            Message *yin, Message *yout,
            bool initialise=true);

        virtual void Initialise();
        virtual void Update(int which=0, const char *debug=0);
        virtual const char* Type() const { return "TRANSFORM"; }

    private:
        const Eigen::MatrixXd m_A;
        Eigen::JacobiSVD<Eigen::MatrixXd> m_svdA;
        MessagePair m_x;
        MessagePair m_y;
    };
}
